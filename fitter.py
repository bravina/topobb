"""
fitter.py — Fit signal strength μ using fiducial + differential cross sections.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from pathlib import Path
from typing import Optional
from observable import Observable
import loader


class FitResult:
    """Container for fit results."""

    def __init__(self, mu, mu_up, mu_down, chi2, ndf,
                 chi2_fid=0.0, chi2_shape=0.0,
                 baseline_label="", observables_used=None):
        self.mu = mu
        self.mu_up = mu_up      # +1σ
        self.mu_down = mu_down  # -1σ
        self.chi2 = chi2
        self.ndf = ndf
        self.chi2_fid = chi2_fid
        self.chi2_shape = chi2_shape
        self.baseline_label = baseline_label
        self.observables_used = observables_used or []

    @property
    def pvalue(self):
        from scipy.stats import chi2 as chi2_dist
        if self.ndf <= 0:
            return -1.0
        return 1.0 - chi2_dist.cdf(self.chi2, self.ndf)

    def __repr__(self):
        obs_str = ", ".join(o.name for o in self.observables_used) or "none"
        return (
            f"FitResult(baseline={self.baseline_label})\n"
            f"  μ = {self.mu:.4f}  +{self.mu_up:.4f}  -{self.mu_down:.4f}\n"
            f"  χ²/ndf = {self.chi2:.2f}/{self.ndf} "
            f"(χ²_fid={self.chi2_fid:.2f}, χ²_shape={self.chi2_shape:.2f})\n"
            f"  p-value = {self.pvalue:.4f}\n"
            f"  Observables: {obs_str}"
        )

    def save(self, filepath: str):
        """Save fit result to text file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(repr(self))
            f.write("\n")


class Fitter:
    """
    Fits signal strength μ for a new-physics signal on top of a SM baseline.

    The χ² has two components:
      1. Fiducial:  (σ_meas - σ_base - μ·σ_sig)² / δσ²
      2. Shape:     (f_meas - f_pred(μ))^T C^{-1} (f_meas - f_pred(μ))
         summed over all included observables, dropping the last bin of each.

    Parameters
    ----------
    fiducial_region : str
        Region name in HEPdata (default: "$\\geq3b$").
    baseline_key : str
        Key to identify the baseline prediction in HEPdata
        (substring match, e.g. "tt (5FS)" for Powheg+Pythia8 tt).
    signal_sigma : float
        Fiducial cross section of the signal process (fb).
    signal_shapes : dict
        {obs_name: np.ndarray} — normalised diff XS of the signal, same
        convention as data (×1000). Must be provided for each Observable used.
    """

    def __init__(
        self,
        baseline_key_fid: str = "tt (5FS)",
        baseline_key_diff: str = "tt (5FS)",
        fiducial_region: str = "$\\geq3b$",
        signal_sigma: float = 0.0,
        signal_shapes: Optional[dict] = None,
    ):
        self.baseline_key_fid = baseline_key_fid
        self.baseline_key_diff = baseline_key_diff
        self.fiducial_region = fiducial_region
        self.signal_sigma = signal_sigma
        self.signal_shapes = signal_shapes or {}

        # Load fiducial data
        self._fid = loader.load_fiducial(region=fiducial_region)
        self.sigma_meas = self._fid["value"]
        self.sigma_stat = self._fid["stat"]
        self.sigma_syst = self._fid["syst"]

        # Baseline fiducial prediction
        self.sigma_baseline = self._get_fiducial_prediction(baseline_key_fid)

        # Total fiducial uncertainty (symmetrised, stat + syst in quadrature)
        self._sigma_total = self._compute_fiducial_total_error()

    def _get_fiducial_prediction(self, key: str) -> float:
        preds = self._fid["predictions"]
        # Try exact then substring
        if key in preds:
            return preds[key]
        for k, v in preds.items():
            if key in k:
                return v
        raise KeyError(f"Fiducial prediction '{key}' not found. Available: {list(preds.keys())}")

    def _compute_fiducial_total_error(self) -> float:
        """Symmetrised total fiducial uncertainty."""
        total_sq = self.sigma_stat ** 2
        for source, (d, u) in self.sigma_syst.items():
            delta = (abs(u) + abs(d)) / 2.0
            total_sq += delta ** 2
        return np.sqrt(total_sq)

    def chi2_fiducial(self, mu: float) -> float:
        """Fiducial cross-section χ² term."""
        pred = self.sigma_baseline + mu * self.signal_sigma
        return ((self.sigma_meas - pred) / self._sigma_total) ** 2

    def chi2_shape(
        self,
        mu: float,
        observables: list,
        use_bootstraps: bool = False,
    ) -> float:
        """
        Shape χ² for one or more observables.

        Drops the last bin of each observable (normalisation constraint).

        Parameters
        ----------
        mu : float
        observables : list of Observable
        use_bootstraps : bool
            If True and multiple observables are given, build the full
            cross-observable covariance from bootstraps (stat) + syst breakdown.
            For a single observable, the provided correlation matrix is used
            regardless.
        """
        if len(observables) == 0:
            return 0.0

        if len(observables) == 1 and not use_bootstraps:
            return self._chi2_shape_single(mu, observables[0])

        # Multiple observables: build combined covariance
        return self._chi2_shape_combined(mu, observables, use_bootstraps)

    def _chi2_shape_single(self, mu: float, obs: Observable) -> float:
        """χ² for a single observable using its provided correlation matrix."""
        # Get signal shape
        sig_shape = self.signal_shapes.get(obs.name)
        if sig_shape is None:
            raise ValueError(f"No signal shape provided for observable '{obs.name}'")

        # Combined prediction
        pred = obs.combined_prediction(
            self.baseline_key_diff, sig_shape,
            self.sigma_baseline, self.signal_sigma, mu,
        )

        # Drop last bin
        n = obs.nbins - 1
        delta = (obs.data[:n] - pred[:n])

        # Covariance from correlation matrix (full), then drop last row/col
        cov_full = obs.covariance
        cov = cov_full[:n, :n]

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        return float(delta @ cov_inv @ delta)

    def _chi2_shape_combined(
        self, mu: float, observables: list, use_bootstraps: bool
    ) -> float:
        """χ² for multiple observables with cross-correlations."""
        # Build combined data and prediction vectors (drop last bin each)
        delta_list = []
        for obs in observables:
            sig_shape = self.signal_shapes.get(obs.name)
            if sig_shape is None:
                raise ValueError(f"No signal shape for '{obs.name}'")
            pred = obs.combined_prediction(
                self.baseline_key_diff, sig_shape,
                self.sigma_baseline, self.signal_sigma, mu,
            )
            n = obs.nbins - 1
            delta_list.append(obs.data[:n] - pred[:n])

        delta = np.concatenate(delta_list)

        # Build combined covariance
        cov = self._build_combined_covariance(observables, use_bootstraps)

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        return float(delta @ cov_inv @ delta)

    def _build_combined_covariance(
        self, observables: list, use_bootstraps: bool
    ) -> np.ndarray:
        """
        Build the combined covariance matrix across multiple observables.

        Statistical correlations: from bootstraps if available and requested,
            otherwise diagonal blocks from per-observable stat errors.
        Systematic correlations: from the named systematic breakdown,
            assuming 100% correlation of each source across observables.
        """
        # Sizes (dropping last bin)
        sizes = [obs.nbins - 1 for obs in observables]
        total = sum(sizes)

        # --- Statistical covariance ---
        cov_stat = np.zeros((total, total))
        if use_bootstraps and all(obs.bootstraps is not None for obs in observables):
            # Stack all bootstraps (drop last bin of each)
            bs_list = []
            for obs in observables:
                bs = obs.bootstraps * obs.scale_factor  # to HEPdata units
                bs_list.append(bs[:, : obs.nbins - 1])
            bs_all = np.hstack(bs_list)
            cov_stat = np.cov(bs_all, rowvar=False)
        else:
            # Block diagonal from individual stat covariances
            offset = 0
            for obs, sz in zip(observables, sizes):
                sc = obs.stat_covariance[:sz, :sz]
                cov_stat[offset : offset + sz, offset : offset + sz] = sc
                offset += sz

        # --- Systematic covariance ---
        # Collect all systematic source names
        all_sources = set()
        for obs in observables:
            all_sources.update(obs.syst_breakdown.keys())

        cov_syst = np.zeros((total, total))
        for source in all_sources:
            delta_vec = []
            for obs in observables:
                n = obs.nbins - 1
                if source in obs.syst_breakdown:
                    down, up = obs.syst_breakdown[source]
                    d = (up[:n] - down[:n]) / 2.0
                else:
                    d = np.zeros(n)
                delta_vec.append(d)
            delta = np.concatenate(delta_vec)
            cov_syst += np.outer(delta, delta)

        return cov_stat + cov_syst

    def fit(
        self,
        observables: Optional[list] = None,
        mu_range: tuple = (-5.0, 10.0),
        use_bootstraps: bool = False,
        include_fiducial: bool = True,
    ) -> FitResult:
        """
        Fit the signal strength μ.

        Parameters
        ----------
        observables : list of Observable, optional
            Differential distributions to include.
        mu_range : tuple
            Search range for μ.
        use_bootstraps : bool
            Use bootstraps for cross-observable stat correlations.
        include_fiducial : bool
            Whether to include the fiducial χ² term.

        Returns
        -------
        FitResult
        """
        observables = observables or []

        def chi2_total(mu):
            c2 = 0.0
            if include_fiducial:
                c2 += self.chi2_fiducial(mu)
            c2 += self.chi2_shape(mu, observables, use_bootstraps)
            return c2

        # Minimize
        result = minimize_scalar(chi2_total, bounds=mu_range, method="bounded")
        mu_best = result.x
        chi2_min = result.fun

        # Δχ² = 1 interval
        mu_down = self._find_delta_chi2(chi2_total, mu_best, chi2_min, direction=-1, mu_range=mu_range)
        mu_up = self._find_delta_chi2(chi2_total, mu_best, chi2_min, direction=+1, mu_range=mu_range)

        # Breakdown
        chi2_f = self.chi2_fiducial(mu_best) if include_fiducial else 0.0
        chi2_s = self.chi2_shape(mu_best, observables, use_bootstraps)

        # NDF: 1 (fiducial, if included) + Σ(nbins-1) per observable - 1 (fitted parameter)
        ndf = (1 if include_fiducial else 0)
        for obs in observables:
            ndf += obs.nbins - 1
        ndf -= 1  # one fitted parameter

        return FitResult(
            mu=mu_best,
            mu_up=mu_up,
            mu_down=mu_down,
            chi2=chi2_min,
            ndf=ndf,
            chi2_fid=chi2_f,
            chi2_shape=chi2_s,
            baseline_label=self.baseline_key_diff,
            observables_used=observables,
        )

    def goodness_of_fit(
        self,
        observables: list,
        mu: float = 0.0,
        use_bootstraps: bool = False,
        include_fiducial: bool = True,
        label: str = "",
    ) -> FitResult:
        """
        Evaluate χ² at a fixed μ (no fit) — useful for hypothesis testing.
        """
        chi2_f = self.chi2_fiducial(mu) if include_fiducial else 0.0
        chi2_s = self.chi2_shape(mu, observables, use_bootstraps)
        chi2_tot = chi2_f + chi2_s

        ndf = (1 if include_fiducial else 0)
        for obs in observables:
            ndf += obs.nbins - 1
        # No free parameter subtracted

        return FitResult(
            mu=mu, mu_up=0.0, mu_down=0.0,
            chi2=chi2_tot, ndf=ndf,
            chi2_fid=chi2_f, chi2_shape=chi2_s,
            baseline_label=label or self.baseline_key_diff,
            observables_used=observables,
        )

    @staticmethod
    def _find_delta_chi2(chi2_func, mu_best, chi2_min, direction, mu_range):
        """Find |mu - mu_best| where Δχ² = 1."""
        from scipy.optimize import brentq

        target = chi2_min + 1.0

        def f(mu):
            return chi2_func(mu) - target

        if direction > 0:
            lo, hi = mu_best, mu_range[1]
        else:
            lo, hi = mu_range[0], mu_best

        # Check that the function crosses zero in the range
        if f(lo) * f(hi) > 0:
            # Δχ² = 1 not reached in range → return boundary distance
            return abs(hi - mu_best) if direction > 0 else abs(mu_best - lo)

        try:
            mu_boundary = brentq(f, lo, hi, xtol=1e-6)
            return abs(mu_boundary - mu_best)
        except ValueError:
            return abs(hi - mu_best) if direction > 0 else abs(mu_best - lo)
