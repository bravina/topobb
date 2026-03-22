"""
signal.py — Load signal predictions from Rivet YODA output.

Parses a YODA file, extracts normalised differential cross sections and
fiducial cross section for a signal process, computes shape-only
scale (muR, muF separate) and PDF uncertainties, and provides the
arrays expected by Fitter, Plotter, and Observable.

Weight naming convention (MadGraph5_aMC@NLO + Pythia8):
    Nominal:       MUR1_MUF1_PDF260400
    muR up:        MUR2_MUF1_PDF260400
    muR down:      MUR0.5_MUF1_PDF260400
    muF up:        MUR1_MUF2_PDF260400
    muF down:      MUR1_MUF0.5_PDF260400
    PDF replicas:  MUR1_MUF1_PDF260401 ... MUR1_MUF1_PDF260500

Usage:
    from signal import SignalLoader

    sig = SignalLoader(
        yoda_path="Rivet.yoda",
        analysis="ATLAS_2024_I2809112",
        observable_map={
            "H__T___had__in__3b": "d71-x01-y01",
        },
        fiducial_histo_id="d01-x01-y01",   # histogram for fiducial XS
    )
    sig.load()

    # Use with the framework
    fitter = Fitter(
        ...,
        signal_sigma=sig.fiducial_xsec,
        signal_shapes=sig.shapes,
    )
"""

import re
import copy
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# =========================================================================
# YODA parsing (lightweight, no yoda Python bindings required)
# =========================================================================

def parse_yoda_file(filepath: str) -> dict:
    """
    Parse a YODA file into a dict of {path: histogram_data}.

    Handles YODA_HISTO1D_V2, YODA_SCATTER2D_V2, BinnedHisto, and
    BinnedEstimate objects (Rivet 3.x and 4.x).
    """
    histograms = {}
    current_path = None
    current_type = None
    current_data = []

    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("BEGIN "):
                parts = line.split()
                current_type = parts[1] if len(parts) > 1 else None
                current_path = parts[2] if len(parts) > 2 else None
                current_data = []
                continue

            if line.startswith("END "):
                if current_path and current_type:
                    utype = current_type.upper()
                    if "HISTO" in utype or "BINNEDH" in utype:
                        histograms[current_path] = _parse_histo_lines(current_data)
                    elif "SCATTER2D" in utype:
                        histograms[current_path] = _parse_scatter2d_lines(current_data)
                    elif "BINNEDE" in utype:
                        histograms[current_path] = _parse_estimate_lines(current_data)
                current_path = None
                current_type = None
                current_data = []
                continue

            current_data.append(line)

    return histograms


def _try_parse_float_line(line):
    """Try to parse a line of space-separated floats. Returns list or None."""
    if line.startswith("#") or line.startswith("---") or ":" in line or "=" in line:
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        return None


def _parse_histo_lines(lines):
    """Parse histogram data: xlow xhigh sumw sumw2 ..."""
    bins = []
    for line in lines:
        vals = _try_parse_float_line(line)
        if vals and len(vals) >= 4:
            bins.append({"xlow": vals[0], "xhigh": vals[1], "sumw": vals[2]})
    return {"type": "histo", "bins": bins}


def _parse_scatter2d_lines(lines):
    """Parse Scatter2D: xval xerr- xerr+ yval yerr- yerr+."""
    points = []
    for line in lines:
        vals = _try_parse_float_line(line)
        if vals and len(vals) >= 6:
            points.append({
                "xlow": vals[0] - vals[1],
                "xhigh": vals[0] + vals[2],
                "y": vals[3],
            })
    return {"type": "scatter2d", "points": points}


def _parse_estimate_lines(lines):
    """Parse BinnedEstimate: xlow xhigh y ..."""
    bins = []
    for line in lines:
        vals = _try_parse_float_line(line)
        if vals and len(vals) >= 3:
            bins.append({"xlow": vals[0], "xhigh": vals[1], "y": vals[2]})
    return {"type": "estimate", "bins": bins}


def _get_bin_values(histo: dict):
    """Extract (edges, values) from a parsed YODA histogram object."""
    if histo["type"] == "histo":
        edges = [(b["xlow"], b["xhigh"]) for b in histo["bins"]]
        values = np.array([b["sumw"] for b in histo["bins"]])
    elif histo["type"] == "scatter2d":
        edges = [(p["xlow"], p["xhigh"]) for p in histo["points"]]
        values = np.array([p["y"] for p in histo["points"]])
    elif histo["type"] == "estimate":
        edges = [(b["xlow"], b["xhigh"]) for b in histo["bins"]]
        values = np.array([b["y"] for b in histo["bins"]])
    else:
        raise ValueError(f"Unknown histogram type: {histo['type']}")
    return edges, values


# =========================================================================
# Variation identification
# =========================================================================

# Standard weight labels
_NOM_LABEL = "MUR1_MUF1_PDF260400"
_MUR_UP    = "MUR2_MUF1_PDF260400"
_MUR_DN    = "MUR0.5_MUF1_PDF260400"
_MUF_UP    = "MUR1_MUF2_PDF260400"
_MUF_DN    = "MUR1_MUF0.5_PDF260400"

# PDF replicas: MUR1_MUF1_PDF260401 .. MUR1_MUF1_PDF260500
_PDF_NOMINAL_ID = 260400
_PDF_REPLICA_RANGE = range(260401, 260501)  # 100 replicas


def _variation_path(base_path: str, label: str) -> str:
    """Construct the YODA path for a weight variation."""
    return f"{base_path}[{label}]"


def _pdf_replica_label(pdf_id: int) -> str:
    return f"MUR1_MUF1_PDF{pdf_id}"


# =========================================================================
# Shape-only uncertainty computation
# =========================================================================

def _renormalise_to(values: np.ndarray, target_integral: float) -> np.ndarray:
    """Renormalise bin values so they integrate to `target_integral`."""
    integral = np.sum(values)
    if integral == 0 or target_integral == 0:
        return values.copy()
    return values * (target_integral / integral)


def compute_shape_envelope(
    nominal: np.ndarray,
    var_up: Optional[np.ndarray],
    var_dn: Optional[np.ndarray],
) -> tuple:
    """
    Compute shape-only uncertainty from a two-point (up/down) variation.

    Each variation is renormalised to the nominal integral, then:
        err_up[i] = max(shift_up[i], shift_dn[i], 0)
        err_dn[i] = min(shift_up[i], shift_dn[i], 0)

    Returns (err_up, err_dn) where err_up ≥ 0 and err_dn ≤ 0.
    """
    n = len(nominal)
    nom_int = np.sum(nominal)

    shifts = []
    if var_up is not None:
        renormed = _renormalise_to(var_up, nom_int)
        shifts.append(renormed - nominal)
    if var_dn is not None:
        renormed = _renormalise_to(var_dn, nom_int)
        shifts.append(renormed - nominal)

    if not shifts:
        return np.zeros(n), np.zeros(n)

    shift_stack = np.array(shifts)  # (n_var, n_bins)
    err_up = np.maximum(np.max(shift_stack, axis=0), 0.0)
    err_dn = np.minimum(np.min(shift_stack, axis=0), 0.0)
    return err_up, err_dn


def compute_pdf_rms(
    nominal: np.ndarray,
    replica_values: list,
) -> tuple:
    """
    Compute shape-only PDF uncertainty as the RMS of replicas.

    Each replica is renormalised to the nominal integral (shape only),
    then the per-bin RMS of (replica - nominal) is computed.

    Returns (err_up, err_dn) both as symmetric ± RMS (err_up ≥ 0, err_dn ≤ 0).
    """
    n = len(nominal)
    if not replica_values:
        return np.zeros(n), np.zeros(n)

    nom_int = np.sum(nominal)
    renormed = np.array([_renormalise_to(r, nom_int) for r in replica_values])
    diffs = renormed - nominal[np.newaxis, :]
    rms = np.sqrt(np.mean(diffs ** 2, axis=0))
    return rms, -rms


# =========================================================================
# SignalPrediction data class
# =========================================================================

@dataclass
class SignalPrediction:
    """
    Holds the signal prediction for all observables, ready for the framework.

    Attributes
    ----------
    fiducial_xsec : float
        Fiducial cross section in fb (before normalisation).
    shapes : dict
        {obs_name: np.ndarray} — normalised differential XS, ×1000 convention.
    uncertainties : dict
        {obs_name: {source: (err_dn_array, err_up_array)}} per bin.
        Sources: "muR", "muF", "PDF".  err_dn ≤ 0, err_up ≥ 0.
    bins : dict
        {obs_name: [(xlow, xhigh), ...]} — bin edges per observable.
    label : str
        Human-readable label.
    """
    fiducial_xsec: float = 0.0
    shapes: dict = field(default_factory=dict)
    uncertainties: dict = field(default_factory=dict)
    bins: dict = field(default_factory=dict)
    label: str = "MG signal"

    def summary(self):
        """Print a summary of the signal prediction."""
        print(f"\n{'='*60}")
        print(f"SIGNAL: {self.label}")
        print(f"{'='*60}")
        print(f"  Fiducial σ = {self.fiducial_xsec:.4f} fb")
        for obs_name, shape in self.shapes.items():
            bw = np.array([h - l for l, h in self.bins[obs_name]])
            integ = np.sum(shape * bw) / 1000.0
            print(f"\n  Observable: {obs_name}")
            print(f"    Bins: {len(shape)}")
            print(f"    Shape: {np.round(shape, 4)}")
            print(f"    Integral check: {integ:.4f} (expect ~1)")
            if obs_name in self.uncertainties:
                for src, (dn, up) in self.uncertainties[obs_name].items():
                    avg_up = np.mean(np.abs(up))
                    avg_dn = np.mean(np.abs(dn))
                    print(f"    {src:5s} shape unc (avg): +{avg_up:.4g} / -{avg_dn:.4g}")

    def get_signal_shapes_dict(self) -> dict:
        """Return the dict expected by Fitter(signal_shapes=...)."""
        return dict(self.shapes)


# =========================================================================
# SignalLoader
# =========================================================================

class SignalLoader:
    """
    Load signal predictions from a Rivet YODA file and prepare them for
    the TopoBB reinterpretation framework.

    Parameters
    ----------
    yoda_path : str
        Path to the Rivet YODA output file.
    analysis : str
        Rivet analysis name (e.g. "ATLAS_2024_I2809112").
    observable_map : dict
        {framework_obs_name: rivet_histo_id} mapping, e.g.
        {"H__T___had__in__3b": "d71-x01-y01"}.
    fiducial_histo_id : str or None
        Rivet histogram ID for the fiducial cross section (if available
        as a normalised histogram whose pre-normalisation integral gives σ_fid).
        If None, fiducial_xsec must be set manually.
    fiducial_xsec : float or None
        If provided, overrides the fiducial cross section instead of
        extracting it from the YODA file.
    scale_factor : float
        The ×1000 factor used in the HEPdata convention (default 1000).
    signal_label : str
        Human-readable label for the signal.
    """

    def __init__(
        self,
        yoda_path: str,
        analysis: str = "ATLAS_2024_I2809112",
        observable_map: Optional[dict] = None,
        fiducial_histo_id: Optional[str] = None,
        fiducial_xsec: Optional[float] = None,
        scale_factor: float = 1000.0,
        signal_label: str = "MG topobb",
    ):
        self.yoda_path = yoda_path
        self.analysis = analysis
        self.observable_map = observable_map or {}
        self.fiducial_histo_id = fiducial_histo_id
        self._fiducial_xsec_override = fiducial_xsec
        self.scale_factor = scale_factor
        self.signal_label = signal_label

        self._histograms = None
        self._prediction = None

    def load(self) -> SignalPrediction:
        """Parse the YODA file and build the SignalPrediction."""
        print(f"Reading YODA file: {self.yoda_path}")
        self._histograms = parse_yoda_file(self.yoda_path)
        print(f"  Found {len(self._histograms)} objects")

        pred = SignalPrediction(label=self.signal_label)

        # --- Fiducial cross section ---
        if self._fiducial_xsec_override is not None:
            pred.fiducial_xsec = self._fiducial_xsec_override
            print(f"  Fiducial σ (user-provided): {pred.fiducial_xsec:.4f} fb")
        elif self.fiducial_histo_id:
            pred.fiducial_xsec = self._extract_fiducial_xsec()
            print(f"  Fiducial σ (from YODA): {pred.fiducial_xsec:.4f} fb")
        else:
            print("  WARNING: no fiducial σ specified — set it manually")

        # --- Per-observable shapes and uncertainties ---
        for obs_name, histo_id in self.observable_map.items():
            print(f"\n  Processing observable: {obs_name} ({histo_id})")
            shape, unc, bins = self._process_observable(histo_id)
            if shape is not None:
                pred.shapes[obs_name] = shape
                pred.uncertainties[obs_name] = unc
                pred.bins[obs_name] = bins

        self._prediction = pred
        return pred

    @property
    def prediction(self) -> SignalPrediction:
        if self._prediction is None:
            raise RuntimeError("Call .load() first")
        return self._prediction

    # -----------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------

    def _base_path(self, histo_id: str) -> str:
        """Construct the YODA base path for a histogram."""
        return f"/{self.analysis}/{histo_id}"

    def _get_histo(self, histo_id: str, variation: Optional[str] = None):
        """Get a histogram from the YODA file, optionally with a variation."""
        base = self._base_path(histo_id)
        if variation:
            path = _variation_path(base, variation)
        else:
            path = base
        return self._histograms.get(path)

    def _get_values(self, histo_id: str, variation: Optional[str] = None):
        """Get (edges, values) for a histogram, optionally with a variation."""
        histo = self._get_histo(histo_id, variation)
        if histo is None:
            return None, None
        return _get_bin_values(histo)

    def _extract_fiducial_xsec(self) -> float:
        """
        Extract the fiducial cross section from a YODA histogram.

        For a normalised histogram, the pre-normalisation integral is the
        fiducial XS.  For a raw histogram, the integral (sumw) gives the
        XS directly (assuming Rivet processed events with proper weighting).
        """
        base = self._base_path(self.fiducial_histo_id)

        # Try the nominal weight variation first
        nom_path = _variation_path(base, _NOM_LABEL)
        if nom_path in self._histograms:
            _, values = _get_bin_values(self._histograms[nom_path])
        elif base in self._histograms:
            _, values = _get_bin_values(self._histograms[base])
        else:
            print(f"    WARNING: fiducial histogram {base} not found")
            return 0.0

        # The integral of the normalised diff XS × bin widths gives 1
        # But we need the absolute XS.  If the histogram is already
        # a normalised dσ/dx, the fiducial XS is lost.
        # → The user should provide fiducial_xsec explicitly or use
        #   a non-normalised histogram (e.g. a counter or raw dσ/dx).
        return float(np.sum(values))

    def _process_observable(self, histo_id: str):
        """
        Extract the normalised differential shape + uncertainties for one
        observable.

        Returns (shape, uncertainties, bins) or (None, None, None).
        """
        # --- Nominal ---
        # Try the explicit nominal weight first, then the bare path
        edges, nominal = self._get_values(histo_id, _NOM_LABEL)
        if nominal is None:
            edges, nominal = self._get_values(histo_id)
        if nominal is None:
            print(f"    WARNING: nominal histogram not found for {histo_id}")
            return None, None, None

        nbins = len(nominal)
        bins = list(edges)
        print(f"    {nbins} bins, nominal integral: {np.sum(nominal):.6g}")

        # The Rivet routine normalises the histogram (it's a normalised dσ/dX).
        # We need it in the ×1000 convention matching HEPdata.
        # If it already integrates to ~1 (in natural units), multiply by scale_factor.
        bin_widths = np.array([h - l for l, h in edges])
        raw_integral = np.sum(nominal * bin_widths)

        # Determine if already normalised (integral ~ 1) or absolute
        if abs(raw_integral - 1.0) < 0.1:
            # Already normalised, just apply scale factor
            shape = nominal * self.scale_factor
            print(f"    Detected normalised histogram (∫f·dx = {raw_integral:.4f})")
        elif abs(raw_integral / self.scale_factor - 1.0) < 0.1:
            # Already in ×1000 convention
            shape = nominal.copy()
            print(f"    Detected ×1000 convention histogram")
        else:
            # Absolute dσ/dx → need to normalise
            total_xs = np.sum(nominal * bin_widths)
            if total_xs > 0:
                shape = (nominal / total_xs) * self.scale_factor
                print(f"    Normalising from absolute (∫ = {total_xs:.4g})")
            else:
                shape = nominal.copy()
                print(f"    WARNING: zero integral, cannot normalise")

        # Verify
        check = np.sum(shape * bin_widths) / self.scale_factor
        print(f"    Shape integral check: {check:.4f} (expect ~1)")

        # --- Scale uncertainties (shape-only) ---
        # muR: MUR2 and MUR0.5 (muF fixed at 1)
        _, mur_up_vals = self._get_values(histo_id, _MUR_UP)
        _, mur_dn_vals = self._get_values(histo_id, _MUR_DN)
        mur_up_shape = self._to_shape(mur_up_vals, bin_widths)
        mur_dn_shape = self._to_shape(mur_dn_vals, bin_widths)

        mur_err_up, mur_err_dn = compute_shape_envelope(shape, mur_up_shape, mur_dn_shape)
        n_mur = sum(1 for v in [mur_up_vals, mur_dn_vals] if v is not None)
        print(f"    muR variations found: {n_mur}/2")

        # muF: MUF2 and MUF0.5 (muR fixed at 1)
        _, muf_up_vals = self._get_values(histo_id, _MUF_UP)
        _, muf_dn_vals = self._get_values(histo_id, _MUF_DN)
        muf_up_shape = self._to_shape(muf_up_vals, bin_widths)
        muf_dn_shape = self._to_shape(muf_dn_vals, bin_widths)

        muf_err_up, muf_err_dn = compute_shape_envelope(shape, muf_up_shape, muf_dn_shape)
        n_muf = sum(1 for v in [muf_up_vals, muf_dn_vals] if v is not None)
        print(f"    muF variations found: {n_muf}/2")

        # --- PDF uncertainty (shape-only RMS of 100 replicas) ---
        pdf_replicas_shape = []
        n_pdf_found = 0
        for pdf_id in _PDF_REPLICA_RANGE:
            label = _pdf_replica_label(pdf_id)
            _, vals = self._get_values(histo_id, label)
            if vals is not None:
                s = self._to_shape(vals, bin_widths)
                if s is not None:
                    pdf_replicas_shape.append(s)
                    n_pdf_found += 1
        print(f"    PDF replicas found: {n_pdf_found}/100")

        pdf_err_up, pdf_err_dn = compute_pdf_rms(shape, pdf_replicas_shape)

        # --- Assemble uncertainties ---
        unc = {
            "muR": (mur_err_dn, mur_err_up),
            "muF": (muf_err_dn, muf_err_up),
            "PDF": (pdf_err_dn, pdf_err_up),
        }

        return shape, unc, bins

    def _to_shape(self, raw_values, bin_widths):
        """
        Convert raw histogram values to the ×1000 normalised shape,
        applying the same normalisation procedure as the nominal.
        Returns None if raw_values is None.
        """
        if raw_values is None:
            return None
        raw_integral = np.sum(raw_values * bin_widths)
        if abs(raw_integral - 1.0) < 0.1:
            return raw_values * self.scale_factor
        elif abs(raw_integral / self.scale_factor - 1.0) < 0.1:
            return raw_values.copy()
        else:
            total_xs = np.sum(raw_values * bin_widths)
            if total_xs > 0:
                return (raw_values / total_xs) * self.scale_factor
            return raw_values.copy()

    # -----------------------------------------------------------------
    # HEPdata injection (optional)
    # -----------------------------------------------------------------

    def inject_into_hepdata(
        self,
        hepdata_dir: str,
        output_dir: str,
        obs_name: str,
    ):
        """
        Inject signal prediction as an extra dependent variable into an
        existing HEPdata YAML file.

        Parameters
        ----------
        hepdata_dir : str
            Directory containing original HEPdata YAML files.
        output_dir : str
            Output directory for modified files.
        obs_name : str
            Observable name (must already be loaded in self._prediction).
        """
        pred = self.prediction
        if obs_name not in pred.shapes:
            raise ValueError(f"Observable '{obs_name}' not loaded")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Find the HEPdata file
        from loader import INPUT_DIR, HEPDATA_PREFIX
        hepdata_path = Path(hepdata_dir) / f"{HEPDATA_PREFIX}Diff__XS_{obs_name}.yaml"
        if not hepdata_path.exists():
            print(f"WARNING: HEPdata file not found: {hepdata_path}")
            return

        with open(hepdata_path) as f:
            hepdata = yaml.safe_load(f)

        hepdata = copy.deepcopy(hepdata)

        shape = pred.shapes[obs_name]
        unc = pred.uncertainties.get(obs_name, {})

        new_dep = {
            "header": {"name": pred.label},
            "values": [],
        }

        for i in range(len(shape)):
            entry = {"value": float(f"{shape[i]:.6g}")}
            errors = []

            for src_name in ["muR", "muF", "PDF"]:
                if src_name in unc:
                    dn, up = unc[src_name]
                    if abs(up[i]) > 1e-10 or abs(dn[i]) > 1e-10:
                        errors.append({
                            "label": f"sys,signal_{src_name}_shape",
                            "asymerror": {
                                "plus": float(f"{up[i]:.6g}"),
                                "minus": float(f"{dn[i]:.6g}"),
                            },
                        })
            if errors:
                entry["errors"] = errors
            new_dep["values"].append(entry)

        hepdata["dependent_variables"].append(new_dep)

        out_path = Path(output_dir) / hepdata_path.name
        with open(out_path, "w") as f:
            yaml.dump(hepdata, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)
        print(f"Injected signal into: {out_path}")

    # -----------------------------------------------------------------
    # CLI
    # -----------------------------------------------------------------

    @classmethod
    def from_cli(cls):
        """Create a SignalLoader from command-line arguments."""
        import argparse
        parser = argparse.ArgumentParser(
            description="Load signal from Rivet YODA and inject into HEPdata",
        )
        parser.add_argument("--yoda", required=True, help="Path to YODA file")
        parser.add_argument("--analysis", default="ATLAS_2024_I2809112")
        parser.add_argument("--fiducial-id", default=None,
                            help="Rivet histogram ID for fiducial XS")
        parser.add_argument("--fiducial-xsec", type=float, default=None,
                            help="Override fiducial XS (fb)")
        parser.add_argument("--signal-label", default="MG topobb")
        parser.add_argument("--hepdata-dir", default="HEPdata_inputs",
                            help="Directory with original HEPdata YAMLs")
        parser.add_argument("--output-dir", default="HEPdata_inputs_with_signal",
                            help="Output directory for modified YAMLs")
        parser.add_argument("--map", nargs=2, action="append", default=[],
                            metavar=("OBS_NAME", "HISTO_ID"),
                            help="Observable mapping (repeatable)")
        args = parser.parse_args()

        obs_map = {name: hid for name, hid in args.map}

        loader = cls(
            yoda_path=args.yoda,
            analysis=args.analysis,
            observable_map=obs_map,
            fiducial_histo_id=args.fiducial_id,
            fiducial_xsec=args.fiducial_xsec,
            signal_label=args.signal_label,
        )
        pred = loader.load()
        pred.summary()

        # Inject into HEPdata if requested
        for obs_name in obs_map:
            loader.inject_into_hepdata(args.hepdata_dir, args.output_dir, obs_name)

        return loader


if __name__ == "__main__":
    SignalLoader.from_cli()
