"""
mapping.py — Central registry of all observables, regions, and name mappings.

Each observable is registered with:
    - short_name : human-friendly key (e.g. "HT_had")
    - region     : jet/b-tag multiplicity bin (e.g. "3j3b")
    - label      : LaTeX label for plots
    - units      : axis units (e.g. "GeV", "" for dimensionless)
    - hepdata_root : file-name root in HEPdata_inputs/ (without _results / _corr_mtrx / _stats)
    - yoda_id    : Rivet histogram identifier (e.g. "d71-x01-y01") — fill in from Rivet routine

The canonical lookup key is (short_name, region), e.g. ("HT_had", "3j3b").

Usage:
    from mapping import get, list_observables, list_regions, get_all_in_region

    info = get("HT_had", "3j3b")
    print(info.label, info.hepdata_root, info.yoda_id)

    for info in get_all_in_region("3j3b"):
        obs = Observable.from_hepdata(info.hepdata_root, info.label, info.units)
"""

from dataclasses import dataclass
from typing import Optional


# =========================================================================
# Data class
# =========================================================================

@dataclass(frozen=True)
class ObsInfo:
    """Metadata for a single observable in a single region."""
    short_name: str
    region: str
    label: str
    units: str
    hepdata_root: str           # e.g. "HT_had_3j3b"
    yoda_id: Optional[str]      # e.g. "d71-x01-y01" (None = not yet mapped)

    @property
    def key(self) -> tuple:
        return (self.short_name, self.region)

    @property
    def full_name(self) -> str:
        """Unique string key: short_name/region."""
        return f"{self.short_name}/{self.region}"

    def __repr__(self):
        yoda = self.yoda_id or "???"
        return f"ObsInfo({self.short_name}/{self.region}  yoda={yoda}  root={self.hepdata_root})"


# =========================================================================
# Region definitions
# =========================================================================

REGIONS = {
    "2j2b": r"$\geq 2j \geq 2b$",
    "3j3b": r"$\geq 3j \geq 3b$",
    "4j4b": r"$\geq 4j \geq 4b$",
    "4j3b": r"$\geq 4j \geq 3b \geq 1l/c$",
    "5j4b": r"$\geq 5j \geq 4b \geq 1l/c$",
}

# Map region short names to the fiducial region keys used in the HEPdata
# fiducial YAML (the header names in the dependent_variables).
FIDUCIAL_REGION_KEYS = {
    "3j3b": r"$\geq3b$",
    "4j4b": r"$\geq4b$",
    "4j3b": r"$\geq3b\geq1l/c$",
    "5j4b": r"$\geq4b\geq1l/c$",
}


# =========================================================================
# Observable definitions — labels and units
# =========================================================================
# Keyed by short_name.  Each observable may exist in multiple regions.

_OBS_DEFS = {
    # --- Kinematics of all b-jets ---
    "bJetPt_1":    (r"$p_{\mathrm{T}}^{b_1}$",                         "GeV"),
    "bJetPt_2":    (r"$p_{\mathrm{T}}^{b_2}$",                         "GeV"),
    "bJetPt_3":    (r"$p_{\mathrm{T}}^{b_3}$",                         "GeV"),
    "bJetPt_4":    (r"$p_{\mathrm{T}}^{b_4}$",                         "GeV"),
    "bJetEta_1":   (r"$|\eta^{b_1}|$",                                  ""),
    "bJetEta_2":   (r"$|\eta^{b_2}|$",                                  ""),
    "bJetEta_3":   (r"$|\eta^{b_3}|$",                                  ""),
    "bJetEta_4":   (r"$|\eta^{b_4}|$",                                  ""),

    # --- Scalar sums ---
    "HT_had":      (r"$H_{\mathrm{T}}^{\mathrm{had}}$",                "GeV"),
    "HT_all":      (r"$H_{\mathrm{T}}^{\mathrm{all}}$",                "GeV"),

    # --- Multiplicities ---
    "NBJets":      (r"$N_{b\text{-jets}}$",                             ""),
    "NlightJets":  (r"$N_{\text{light jets}}$",                         ""),

    # --- b-jet pair observables ---
    "b1b2JetPt":   (r"$p_{\mathrm{T}}^{b_1 b_2}$",                     "GeV"),
    "DeltaR_b1b2": (r"$\Delta R_{b_1 b_2}$",                           ""),
    "dRbb_avg":    (r"$\langle \Delta R_{bb} \rangle$",                 ""),
    "Mass_b1b2":   (r"$m_{b_1 b_2}$",                                   "GeV"),
    "Mass_emubb":  (r"$m_{e\mu bb}$",                                   "GeV"),
    "DeltaR_emubb_ab1": (r"$\Delta R_{e\mu bb, \mathrm{add.\,}b_1}$",  ""),
    "max_deltaeta_jj": (r"$\max\Delta\eta_{jj}$",                      ""),

    # --- ≥4b only ---
    "bb_mindR_jetPt":  (r"$p_{\mathrm{T}}^{bb,\min\Delta R}$",         "GeV"),
    "Mass_mindR_bb":   (r"$m_{bb}^{\min\Delta R}$",                     "GeV"),
    "minDeltaR_bjets": (r"$\min\Delta R_{bb}$",                         ""),

    # --- Top-algo reconstructed ---
    "bTopJetPt_1_TopAlgo":  (r"$p_{\mathrm{T}}^{b_{\mathrm{top},1}}$", "GeV"),
    "bTopJetPt_2_TopAlgo":  (r"$p_{\mathrm{T}}^{b_{\mathrm{top},2}}$", "GeV"),
    "bTopJetEta_1_TopAlgo": (r"$|\eta^{b_{\mathrm{top},1}}|$",         ""),
    "bTopJetEta_2_TopAlgo": (r"$|\eta^{b_{\mathrm{top},2}}|$",         ""),
    "AddBJetPt_1_TopAlgo":  (r"$p_{\mathrm{T}}^{\mathrm{add.\,}b_1}$", "GeV"),
    "AddBJetPt_2_TopAlgo":  (r"$p_{\mathrm{T}}^{\mathrm{add.\,}b_2}$", "GeV"),
    "AddBJetEta_1_TopAlgo": (r"$|\eta^{\mathrm{add.\,}b_1}|$",         ""),
    "AddBJetEta_2_TopAlgo": (r"$|\eta^{\mathrm{add.\,}b_2}|$",         ""),
    "bTop1Topb2jetPt_TopAlgo": (r"$p_{\mathrm{T}}^{b_{\mathrm{top},1} b_{\mathrm{top},2}}$", "GeV"),
    "Mass_bTop1bTop2_TopAlgo": (r"$m_{b_{\mathrm{top},1} b_{\mathrm{top},2}}$", "GeV"),
    "Mass_emubb_TopAlgo":     (r"$m_{e\mu bb}^{\mathrm{top\,algo}}$",   "GeV"),
    "DeltaR_emubb_ab1_TopAlgo": (r"$\Delta R_{e\mu bb, \mathrm{add.\,}b_1}^{\mathrm{top\,algo}}$", ""),
    "addb1addb2jetPt_TopAlgo":  (r"$p_{\mathrm{T}}^{\mathrm{add.\,}b_1 \mathrm{add.\,}b_2}$", "GeV"),
    "Mass_ab1ab2_TopAlgo":      (r"$m_{\mathrm{add.\,}b_1 \mathrm{add.\,}b_2}$", "GeV"),

    # --- Light-jet observables (≥1 l/c regions) ---
    "lightJetPt_1":  (r"$p_{\mathrm{T}}^{l/c\text{-jet}_1}$",  "GeV"),
    "lightJetEta_1": (r"$|\eta^{l/c\text{-jet}_1}|$",           ""),
    "dR_emubb_llj":  (r"$\Delta R_{e\mu bb, l/c\text{-jet}}$",  ""),
    "Diff_lljPt_AddB1Pt": (r"$p_{\mathrm{T}}^{l/c\text{-jet}} - p_{\mathrm{T}}^{\mathrm{add.\,}b_1}$", "GeV"),
}


# =========================================================================
# Which observables exist in which regions (from the HEPdata file listing)
# =========================================================================

_OBS_REGIONS = {
    "bJetPt_1":     ["3j3b", "4j4b"],
    "bJetPt_2":     ["3j3b", "4j4b"],
    "bJetPt_3":     ["3j3b", "4j4b"],
    "bJetPt_4":     ["4j4b"],
    "bJetEta_1":    ["3j3b", "4j4b"],
    "bJetEta_2":    ["3j3b", "4j4b"],
    "bJetEta_3":    ["3j3b", "4j4b"],
    "bJetEta_4":    ["4j4b"],
    "HT_had":       ["3j3b", "4j4b"],
    "HT_all":       ["3j3b", "4j4b"],
    "NBJets":       ["2j2b", "3j3b"],
    "NlightJets":   ["3j3b", "4j4b"],
    "b1b2JetPt":    ["3j3b", "4j4b"],
    "DeltaR_b1b2":  ["3j3b", "4j4b"],
    "dRbb_avg":     ["3j3b", "4j4b"],
    "Mass_b1b2":    ["3j3b", "4j4b"],
    "Mass_emubb":   ["3j3b", "4j4b"],
    "DeltaR_emubb_ab1":    ["3j3b", "4j4b"],
    "max_deltaeta_jj":     ["3j3b", "4j4b"],
    "bb_mindR_jetPt":      ["4j4b"],
    "Mass_mindR_bb":       ["4j4b"],
    "minDeltaR_bjets":     ["4j4b"],
    "bTopJetPt_1_TopAlgo":  ["3j3b", "4j4b"],
    "bTopJetPt_2_TopAlgo":  ["3j3b", "4j4b"],
    "bTopJetEta_1_TopAlgo": ["3j3b", "4j4b"],
    "bTopJetEta_2_TopAlgo": ["3j3b", "4j4b"],
    "AddBJetPt_1_TopAlgo":  ["3j3b", "4j4b"],
    "AddBJetPt_2_TopAlgo":  ["4j4b"],
    "AddBJetEta_1_TopAlgo": ["3j3b", "4j4b"],
    "AddBJetEta_2_TopAlgo": ["4j4b"],
    "bTop1Topb2jetPt_TopAlgo": ["3j3b", "4j4b"],
    "Mass_bTop1bTop2_TopAlgo": ["3j3b", "4j4b"],
    "Mass_emubb_TopAlgo":      ["3j3b", "4j4b"],
    "DeltaR_emubb_ab1_TopAlgo": ["3j3b", "4j4b"],
    "addb1addb2jetPt_TopAlgo":  ["4j4b"],
    "Mass_ab1ab2_TopAlgo":      ["4j4b"],
    "lightJetPt_1":         ["4j3b", "5j4b"],
    "lightJetEta_1":        ["4j3b", "5j4b"],
    "dR_emubb_llj":         ["4j3b", "5j4b"],
    "Diff_lljPt_AddB1Pt":   ["4j3b", "5j4b"],
}


# =========================================================================
# HEPdata file-name overrides for case inconsistencies
# =========================================================================
# By default, the root is "{short_name}_{region}".
# Some files have case mismatches between _results / _corr_mtrx / _stats.
# The overrides below specify the file roots per suffix.

_HEPDATA_FILE_OVERRIDES = {
    # b1b2JetPt: _results uses lowercase 'j', _corr_mtrx/_stats use uppercase 'J'
    ("b1b2JetPt", "3j3b"): {
        "results":   "b1b2jetPt_3j3b",
        "corr_mtrx": "b1b2JetPt_3j3b",
        "stats":     "b1b2JetPt_3j3b",
    },
    ("b1b2JetPt", "4j4b"): {
        "results":   "b1b2jetPt_4j4b",
        "corr_mtrx": "b1b2JetPt_4j4b",
        "stats":     "b1b2JetPt_4j4b",
    },
    # DeltaR_emubb_ab1 without TopAlgo: no _results file exists (only corr_mtrx + stats)
    # (these appear in the listing but may be incomplete — flag them)
}


# =========================================================================
# YODA histogram ID mapping
# =========================================================================
# Fill this in from your Rivet routine.
# Format: (short_name, region) -> "dNN-x01-y01"
# TODO: populate from the Rivet analysis source or YODA file header.

_YODA_IDS = {
    # Example (fill in the rest from your Rivet routine):
    # ("HT_had", "3j3b"):  "d71-x01-y01",
    # ("HT_had", "4j4b"):  "d72-x01-y01",
    # ("HT_all", "3j3b"):  "d73-x01-y01",
    # ...
}


# =========================================================================
# Build the registry
# =========================================================================

_REGISTRY: dict[tuple, ObsInfo] = {}


def _build_registry():
    """Populate the registry from the definitions above."""
    for short_name, regions in _OBS_REGIONS.items():
        if short_name not in _OBS_DEFS:
            raise ValueError(f"Observable '{short_name}' listed in _OBS_REGIONS "
                             f"but not defined in _OBS_DEFS")
        label, units = _OBS_DEFS[short_name]
        for region in regions:
            hepdata_root = f"{short_name}_{region}"
            yoda_id = _YODA_IDS.get((short_name, region))
            info = ObsInfo(
                short_name=short_name,
                region=region,
                label=label,
                units=units,
                hepdata_root=hepdata_root,
                yoda_id=yoda_id,
            )
            _REGISTRY[(short_name, region)] = info


_build_registry()


# =========================================================================
# Public API
# =========================================================================

def get(short_name: str, region: str) -> ObsInfo:
    """Look up an observable by (short_name, region)."""
    key = (short_name, region)
    if key not in _REGISTRY:
        raise KeyError(f"Observable {key} not found. "
                       f"Use list_observables() to see available entries.")
    return _REGISTRY[key]


def get_all_in_region(region: str) -> list[ObsInfo]:
    """Return all observables registered in a given region."""
    return [info for (_, r), info in sorted(_REGISTRY.items()) if r == region]


def get_all_for_observable(short_name: str) -> list[ObsInfo]:
    """Return all regions for a given observable."""
    return [info for (n, _), info in sorted(_REGISTRY.items()) if n == short_name]


def list_observables() -> list[str]:
    """Return sorted list of unique short_names."""
    return sorted(set(n for n, _ in _REGISTRY))


def list_regions() -> list[str]:
    """Return sorted list of regions."""
    return sorted(REGIONS.keys())


def get_fiducial_key(region: str) -> str:
    """Return the HEPdata fiducial dependent_variable key for a region."""
    if region not in FIDUCIAL_REGION_KEYS:
        raise KeyError(f"No fiducial key for region '{region}'. "
                       f"Available: {list(FIDUCIAL_REGION_KEYS.keys())}")
    return FIDUCIAL_REGION_KEYS[region]


def get_hepdata_filenames(short_name: str, region: str) -> dict:
    """
    Return the HEPdata file names for an observable.

    Returns dict with keys "results", "corr_mtrx", "stats", each mapping
    to the filename (without directory, with .yaml extension).
    """
    key = (short_name, region)
    overrides = _HEPDATA_FILE_OVERRIDES.get(key, {})
    default_root = f"{short_name}_{region}"

    return {
        suffix: f"{overrides.get(suffix, default_root)}_{suffix}.yaml"
        for suffix in ("results", "corr_mtrx", "stats")
    }


def get_signal_map(region: str) -> dict:
    """
    Return {hepdata_root: yoda_id} for all observables with a YODA mapping
    in the given region.  Ready to pass to SignalLoader(observable_map=...).
    """
    result = {}
    for info in get_all_in_region(region):
        if info.yoda_id:
            result[info.hepdata_root] = info.yoda_id
    return result


def summary():
    """Print a summary of all registered observables."""
    print(f"\n{'='*80}")
    print(f"OBSERVABLE REGISTRY: {len(_REGISTRY)} entries, "
          f"{len(list_observables())} observables, {len(REGIONS)} regions")
    print(f"{'='*80}")
    for region in sorted(REGIONS):
        entries = get_all_in_region(region)
        if not entries:
            continue
        n_mapped = sum(1 for e in entries if e.yoda_id)
        print(f"\n  Region {region} ({REGIONS[region]}): "
              f"{len(entries)} observables, {n_mapped} with YODA mapping")
        for info in entries:
            yoda = info.yoda_id or "---"
            print(f"    {info.short_name:35s} {yoda:20s} {info.label}")


# =========================================================================
# Convenience: load an Observable directly from the mapping
# =========================================================================

def load_observable(short_name: str, region: str):
    """
    Create an Observable instance from the mapping, using the updated loader.

    Returns an Observable ready for fitting/plotting.
    """
    from observable import Observable
    info = get(short_name, region)
    return Observable.from_hepdata(
        name=info.hepdata_root,
        label=info.label,
        units=info.units,
    )


if __name__ == "__main__":
    summary()
