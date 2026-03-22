"""
loader.py — Parse HEPdata YAML files for fiducial and differential cross sections.

File naming convention (from mapping.py):
    {obs_root}_stats.yaml       — differential cross section with uncertainties + MC predictions
    {obs_root}_corr_mtrx.yaml   — correlation matrix
    {obs_root}_results.yaml     — bootstrap replicas (bins as dep. var. headers, replicas as values)
    fid_xsec_systematics.yaml   — fiducial cross sections

The obs_name parameter is the HEPdata file root (e.g. "HT_had_3j3b"),
obtained from mapping.get(short_name, region).hepdata_root.
"""

import yaml
import numpy as np
from pathlib import Path

INPUT_DIR = Path("HEPdata_inputs")
FIDUCIAL_FILENAME = "fid_xsec_systematics.yaml"


# =========================================================================
# Path resolution
# =========================================================================

def _resolve_path(obs_name: str, suffix: str) -> Path:
    """
    Find the HEPdata file for an observable.

    Parameters
    ----------
    obs_name : str
        Observable root, e.g. "HT_had_3j3b".
    suffix : str
        One of "results", "corr_mtrx", "stats".
    """
    # Check for per-file overrides from the mapping module (case mismatches)
    try:
        from mapping import _HEPDATA_FILE_OVERRIDES, REGIONS
        parts = obs_name.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in REGIONS:
            key = (parts[0], parts[1])
            overrides = _HEPDATA_FILE_OVERRIDES.get(key, {})
            if suffix in overrides:
                p = INPUT_DIR / f"{overrides[suffix]}_{suffix}.yaml"
                if p.exists():
                    return p
    except ImportError:
        pass

    # Standard path
    p = INPUT_DIR / f"{obs_name}_{suffix}.yaml"
    if p.exists():
        return p

    # Case-insensitive fallback
    if INPUT_DIR.exists():
        target_lower = f"{obs_name}_{suffix}.yaml".lower()
        for f in INPUT_DIR.iterdir():
            if f.name.lower() == target_lower:
                return f

    raise FileNotFoundError(
        f"No HEPdata file found for obs='{obs_name}', suffix='{suffix}'. "
        f"Expected: {p}"
    )


# =========================================================================
# Error parsing
# =========================================================================

def _parse_error_value(val):
    """Convert a HEPdata error value to float. Empty strings → 0."""
    if val == "" or val is None:
        return 0.0
    return float(val)


def _parse_errors(error_list):
    """
    Parse the error list for a single measured value.

    Returns
    -------
    stat : float
        Symmetric statistical error.
    syst_dict : dict
        {source_name: (down, up)}.
    """
    stat = 0.0
    syst_dict = {}
    for entry in error_list:
        label = entry.get("label", "")
        if label == "stat. error" and "symerror" in entry:
            stat = float(entry["symerror"])
        elif label.startswith("sys,"):
            source = label.replace("sys,", "")
            ae = entry["asymerror"]
            down = _parse_error_value(ae["minus"])
            up = _parse_error_value(ae["plus"])
            syst_dict[source] = (down, up)
    return stat, syst_dict


# =========================================================================
# Fiducial cross section
# =========================================================================

def load_fiducial(region="$\\geq3b$"):
    """
    Load fiducial cross-section results.

    Parameters
    ----------
    region : str
        Region key as it appears in the HEPdata YAML header.
        Use mapping.get_fiducial_key(region_short) to convert from "3j3b" etc.

    Returns
    -------
    dict with keys:
        'value'       : float (measured σ_fid in fb)
        'stat'        : float (stat error)
        'syst'        : dict {source: (down, up)}
        'predictions' : dict {generator_label: float}
    """
    fpath = INPUT_DIR / FIDUCIAL_FILENAME
    if not fpath.exists():
        raise FileNotFoundError(f"Fiducial file not found: {fpath}")

    with open(fpath) as f:
        data = yaml.safe_load(f)

    row_labels = [v["value"] for v in data["independent_variables"][0]["values"]]

    dep = None
    for d in data["dependent_variables"]:
        if d["header"]["name"] == region:
            dep = d
            break
    if dep is None:
        available = [d["header"]["name"] for d in data["dependent_variables"]]
        raise ValueError(f"Region '{region}' not found. Available: {available}")

    values = dep["values"]
    measured_entry = values[0]
    stat, syst = _parse_errors(measured_entry["errors"])

    predictions = {}
    for label, entry in zip(row_labels[1:], values[1:]):
        v = entry.get("value", entry) if isinstance(entry, dict) else entry
        if isinstance(v, dict):
            v = v.get("value")
        if v == "-" or v is None:
            continue
        predictions[label] = float(v)

    return {
        "value": float(measured_entry["value"]),
        "stat": stat,
        "syst": syst,
        "predictions": predictions,
    }


# =========================================================================
# Differential cross section  (reads _stats.yaml)
# =========================================================================

def load_differential(obs_name: str):
    """
    Load normalised differential cross-section from HEPdata.

    Reads {obs_name}_stats.yaml which contains:
      - independent_variables: bin edges
      - dependent_variables[0]: measured data with stat + syst errors
      - dependent_variables[1:]: MC predictions (central values only)

    Parameters
    ----------
    obs_name : str
        Observable root, e.g. "HT_had_3j3b".
    """
    fpath = _resolve_path(obs_name, "stats")
    with open(fpath) as f:
        raw = yaml.safe_load(f)

    # Parse bins from independent variables
    bins = []
    for v in raw["independent_variables"][0]["values"]:
        if "low" in v and "high" in v:
            bins.append((float(v["low"]), float(v["high"])))
        else:
            label = str(v["value"])
            num = "".join(c for c in label if c.isdigit() or c == ".")
            if num:
                bins.append((float(num), np.inf))
    nbins = len(bins)

    # Find data (has errors) and predictions (no errors)
    deps = raw["dependent_variables"]
    data_dep = None
    prediction_deps = []

    for dep in deps:
        values = dep["values"]
        if not values or len(values) != nbins:
            continue
        if "errors" in values[0]:
            data_dep = dep
        else:
            prediction_deps.append(dep)

    if data_dep is None:
        raise ValueError(
            f"No measured data found in {fpath}. "
            f"Dependent variables: {[d['header'] for d in deps]}"
        )

    # Parse measured data
    data_vals = []
    stat_vals = []
    syst_all = {}

    for entry in data_dep["values"]:
        data_vals.append(float(entry["value"]))
        if "errors" in entry:
            st, sy = _parse_errors(entry["errors"])
        else:
            st, sy = 0.0, {}
        stat_vals.append(st)
        for source, (d, u) in sy.items():
            if source not in syst_all:
                syst_all[source] = ([], [])
            syst_all[source][0].append(d)
            syst_all[source][1].append(u)

    syst = {}
    for source, (dlist, ulist) in syst_all.items():
        while len(dlist) < nbins:
            dlist.append(0.0)
        while len(ulist) < nbins:
            ulist.append(0.0)
        syst[source] = (np.array(dlist), np.array(ulist))

    # Parse MC predictions
    predictions = {}
    for dep in prediction_deps:
        name = dep["header"]["name"]
        vals = np.array([float(v["value"]) for v in dep["values"]])
        predictions[name] = vals

    return {
        "bins": bins,
        "data": np.array(data_vals),
        "stat": np.array(stat_vals),
        "syst": syst,
        "predictions": predictions,
        "scale_factor": 1000.0,
    }


# =========================================================================
# Correlation matrix  (reads _corr_mtrx.yaml)
# =========================================================================

def load_correlation(obs_name: str, nbins: int):
    """Load correlation matrix. Returns np.ndarray of shape (nbins, nbins)."""
    fpath = _resolve_path(obs_name, "corr_mtrx")
    with open(fpath) as f:
        raw = yaml.safe_load(f)

    flat = [v["value"] for v in raw["dependent_variables"][0]["values"]]
    corr = np.array(flat).reshape(nbins, nbins)
    return corr


# =========================================================================
# Bootstraps  (reads _results.yaml)
# =========================================================================

def load_bootstraps(obs_name: str):
    """
    Load bootstrap replicas from {obs_name}_results.yaml.

    File structure:
        independent_variables: [{header: {name: Replica}, values: [000, 001, ...]}]
        dependent_variables:
          - {header: {name: "50.0-225.0", units: GeV}, values: [{value: ...}, ...]}
          - {header: {name: "225.0-285.0", units: GeV}, values: [{value: ...}, ...]}
          ...
    Each dependent variable is one bin; each value entry is one replica.

    Returns np.ndarray of shape (n_replicas, n_bins) in NATURAL units (not ×1000).
    """
    fpath = _resolve_path(obs_name, "results")
    with open(fpath) as f:
        raw = yaml.safe_load(f)

    deps = raw["dependent_variables"]
    n_replicas = len(deps[0]["values"])
    n_bins = len(deps)

    bootstraps = np.zeros((n_replicas, n_bins))
    for j, dep in enumerate(deps):
        for i, v in enumerate(dep["values"]):
            bootstraps[i, j] = float(v["value"])

    return bootstraps
