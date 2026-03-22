"""
loader.py — Parse HEPdata YAML files for fiducial and differential cross sections.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Optional

INPUT_DIR = Path("HEPdata_inputs")
HEPDATA_PREFIX = "HEPData-ins2809112-v3-"


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
        {source_name: (down, up)} where down ≤ 0 and up ≥ 0 by convention,
        but stored as given (signed).
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


def load_fiducial(region="$\\geq3b$"):
    """
    Load fiducial cross-section results.

    Returns
    -------
    dict with keys:
        'value'       : float (measured σ_fid in fb)
        'stat'        : float (stat error)
        'syst'        : dict {source: (down, up)}
        'predictions' : dict {generator_label: float}
    """
    fpath = INPUT_DIR / f"{HEPDATA_PREFIX}Fiducial_xsec_results.yaml"
    with open(fpath) as f:
        data = yaml.safe_load(f)

    # Independent variable gives the row labels (Measured, then MC generators)
    row_labels = [v["value"] for v in data["independent_variables"][0]["values"]]

    # Find the dependent variable matching the requested region
    dep = None
    for d in data["dependent_variables"]:
        if d["header"]["name"] == region:
            dep = d
            break
    if dep is None:
        available = [d["header"]["name"] for d in data["dependent_variables"]]
        raise ValueError(f"Region '{region}' not found. Available: {available}")

    values = dep["values"]

    # First entry is the measured value with errors
    measured_entry = values[0]
    stat, syst = _parse_errors(measured_entry["errors"])

    # Remaining entries are MC predictions
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


def load_differential(obs_name: str):
    """
    Load normalised differential cross-section from HEPdata.

    Parameters
    ----------
    obs_name : str
        Observable file identifier, e.g. "H__T___had__in__3b".

    Returns
    -------
    dict with keys:
        'bins'        : list of (low, high) tuples
        'data'        : np.array of measured values (×1000 convention as in HEPdata)
        'stat'        : np.array of stat errors
        'syst'        : dict {source: (np.array_down, np.array_up)} per bin
        'predictions' : dict {generator_label: np.array}
        'scale_factor': float — the multiplicative factor applied in HEPdata (1000)
    """
    fpath = INPUT_DIR / f"{HEPDATA_PREFIX}Diff__XS_{obs_name}.yaml"
    with open(fpath) as f:
        raw = yaml.safe_load(f)

    # Parse bins from independent variables
    bins = []
    for v in raw["independent_variables"][0]["values"]:
        if "low" in v and "high" in v:
            bins.append((float(v["low"]), float(v["high"])))
        else:
            # Overflow bin like '≥540' — store as (low, inf)
            label = str(v["value"])
            # Try to extract number from strings like '≥540'
            num = "".join(c for c in label if c.isdigit() or c == ".")
            bins.append((float(num), np.inf))
    nbins = len(bins)

    # Parse dependent variables
    deps = raw["dependent_variables"]
    # First dep var is data with errors
    data_dep = deps[0]
    data_vals = []
    stat_vals = []
    syst_all = {}  # source -> (list_down, list_up)

    for entry in data_dep["values"]:
        data_vals.append(float(entry["value"]))
        st, sy = _parse_errors(entry["errors"])
        stat_vals.append(st)
        for source, (d, u) in sy.items():
            if source not in syst_all:
                syst_all[source] = ([], [])
            syst_all[source][0].append(d)
            syst_all[source][1].append(u)

    # Convert syst to numpy
    syst = {}
    for source, (dlist, ulist) in syst_all.items():
        # Pad if some bins don't have this source (shouldn't happen, but safety)
        while len(dlist) < nbins:
            dlist.append(0.0)
        while len(ulist) < nbins:
            ulist.append(0.0)
        syst[source] = (np.array(dlist), np.array(ulist))

    # Parse MC predictions (remaining dependent variables)
    predictions = {}
    for dep in deps[1:]:
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


def load_correlation(obs_name: str, nbins: int):
    """
    Load correlation matrix from HEPdata.

    Returns
    -------
    np.ndarray of shape (nbins, nbins)
    """
    fpath = INPUT_DIR / f"{HEPDATA_PREFIX}Corr__mtrx_{obs_name}.yaml"
    with open(fpath) as f:
        raw = yaml.safe_load(f)

    flat = [v["value"] for v in raw["dependent_variables"][0]["values"]]
    corr = np.array(flat).reshape(nbins, nbins)
    return corr


def load_bootstraps(obs_name: str):
    """
    Load bootstrap replicas.

    Returns
    -------
    np.ndarray of shape (n_replicas, n_bins)
        Values in NATURAL units (1/GeV), NOT ×1000.
    """
    fpath = INPUT_DIR / f"{HEPDATA_PREFIX}Bootstrap_{obs_name}.yaml"
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
