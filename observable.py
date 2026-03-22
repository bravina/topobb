"""
observable.py — Observable class representing a normalised differential distribution.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import loader


@dataclass
class Observable:
    """
    A normalised differential cross-section observable.

    All values (data, errors, predictions) are stored in HEPdata convention
    (i.e. with the ×1000 factor if present).
    """

    # Identification
    name: str  # HEPdata file root, e.g. "HT_had_3j3b"
    label: str  # LaTeX label for plotting, e.g. r"$H_{\mathrm{T}}^{\mathrm{had}}$"
    units: str = "GeV"

    # Bin structure
    bins: list = field(default_factory=list)  # [(low, high), ...]

    # Data
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    stat_errors: np.ndarray = field(default_factory=lambda: np.array([]))

    # Full systematic breakdown: {source: (down_array, up_array)}
    syst_breakdown: dict = field(default_factory=dict)

    # MC predictions: {label: np.ndarray}
    predictions: dict = field(default_factory=dict)

    # Correlation and covariance
    correlation: Optional[np.ndarray] = None  # (N, N)
    _covariance: Optional[np.ndarray] = None  # cached
    _stat_covariance: Optional[np.ndarray] = None  # from bootstraps

    # Bootstraps: (n_replicas, n_bins)
    bootstraps: Optional[np.ndarray] = None

    # Scale factor (1000 for ×1000 convention)
    scale_factor: float = 1000.0

    # -------------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------------

    @property
    def nbins(self) -> int:
        return len(self.bins)

    @property
    def bin_widths(self) -> np.ndarray:
        return np.array([h - l for l, h in self.bins])

    @property
    def bin_centres(self) -> np.ndarray:
        return np.array([(l + h) / 2.0 for l, h in self.bins])

    @property
    def bin_edges(self) -> np.ndarray:
        """Array of N+1 bin edges (uses last bin high even if inf → replaced)."""
        edges = [self.bins[0][0]]
        for _, h in self.bins:
            edges.append(h)
        return np.array(edges)

    @property
    def syst_up(self) -> np.ndarray:
        """Total systematic error (up), summed in quadrature."""
        total = np.zeros(self.nbins)
        for source, (down, up) in self.syst_breakdown.items():
            total += np.maximum(up, 0) ** 2
            total += np.maximum(-down, 0) ** 2  # one-sided: if down is positive, it's an upward shift
        # Proper handling: for each source, take max(|up|,0)^2 contribution upward
        # Actually, let's do it properly with symmetrised errors
        return self._total_syst_signed(sign=+1)

    @property
    def syst_down(self) -> np.ndarray:
        """Total systematic error (down), summed in quadrature (returned as positive)."""
        return self._total_syst_signed(sign=-1)

    def _total_syst_signed(self, sign=+1) -> np.ndarray:
        """
        Compute total syst error in one direction.
        For each source, take the shift in the requested direction.
        sign=+1 → upward variations; sign=-1 → downward variations.
        Returns positive values.
        """
        total_sq = np.zeros(self.nbins)
        for source, (down, up) in self.syst_breakdown.items():
            if sign > 0:
                # Upward shift: positive 'up' values contribute
                shift = np.where(up > 0, up, np.where(down > 0, down, 0.0))
            else:
                # Downward shift: negative values contribute
                shift = np.where(up < 0, up, np.where(down < 0, down, 0.0))
            total_sq += shift ** 2
        return np.sqrt(total_sq)

    @property
    def syst_symmetrised(self) -> np.ndarray:
        """Symmetrised total systematic uncertainty per bin."""
        return (self.syst_up + self.syst_down) / 2.0

    @property
    def total_error(self) -> np.ndarray:
        """Total (stat+syst) symmetrised uncertainty per bin."""
        return np.sqrt(self.stat_errors ** 2 + self.syst_symmetrised ** 2)

    @property
    def covariance(self) -> np.ndarray:
        """
        Full covariance matrix from provided correlation + total errors.
        """
        if self._covariance is not None:
            return self._covariance
        if self.correlation is None:
            raise ValueError("No correlation matrix available.")
        sigma = self.total_error
        cov = self.correlation * np.outer(sigma, sigma)
        return cov

    @covariance.setter
    def covariance(self, value):
        self._covariance = value

    @property
    def stat_covariance(self) -> np.ndarray:
        """Statistical covariance from bootstraps."""
        if self._stat_covariance is not None:
            return self._stat_covariance
        if self.bootstraps is not None:
            # Bootstraps are in natural units; scale to match data convention
            bs = self.bootstraps * self.scale_factor
            self._stat_covariance = np.cov(bs, rowvar=False)
            return self._stat_covariance
        # Fallback: diagonal from stat errors
        return np.diag(self.stat_errors ** 2)

    @property
    def syst_covariance(self) -> np.ndarray:
        """
        Systematic covariance from the breakdown.
        Each named source is 100% correlated across bins:
            Cov_syst = Σ_k δ_k δ_k^T
        where δ_k is the symmetrised shift vector for source k.
        """
        n = self.nbins
        cov = np.zeros((n, n))
        for source, (down, up) in self.syst_breakdown.items():
            # Symmetrise: δ = (up - down) / 2  (down is typically negative)
            delta = (up - down) / 2.0
            cov += np.outer(delta, delta)
        return cov

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------

    @classmethod
    def from_hepdata(cls, name: str, label: str, units: str = "GeV"):
        """Load an observable from HEPdata YAML files."""
        diff = loader.load_differential(name)
        nbins = len(diff["bins"])
        corr = loader.load_correlation(name, nbins)
        boots = loader.load_bootstraps(name)

        obs = cls(
            name=name,
            label=label,
            units=units,
            bins=diff["bins"],
            data=diff["data"],
            stat_errors=diff["stat"],
            syst_breakdown=diff["syst"],
            predictions=diff["predictions"],
            correlation=corr,
            bootstraps=boots,
            scale_factor=diff["scale_factor"],
        )
        return obs

    # -------------------------------------------------------------------------
    # Prediction helpers
    # -------------------------------------------------------------------------

    def get_prediction(self, key: str) -> np.ndarray:
        """Get a prediction by key (exact or substring match)."""
        if key in self.predictions:
            return self.predictions[key]
        for k, v in self.predictions.items():
            if key in k:
                return v
        raise KeyError(f"Prediction '{key}' not found. Available: {list(self.predictions.keys())}")

    def combined_prediction(
        self,
        baseline_key: str,
        signal_shape: np.ndarray,
        sigma_baseline: float,
        sigma_signal: float,
        mu: float = 1.0,
    ) -> np.ndarray:
        """
        Compute the normalised differential prediction for baseline + μ × signal.

        Parameters
        ----------
        baseline_key : str
            Key for the baseline prediction in self.predictions.
        signal_shape : np.ndarray
            Normalised differential cross section of the signal (same ×1000 convention).
        sigma_baseline : float
            Fiducial cross section of the baseline (fb).
        sigma_signal : float
            Fiducial cross section of the signal (fb).
        mu : float
            Signal strength.

        Returns
        -------
        np.ndarray : combined normalised prediction in same convention as data.
        """
        f_base = self.get_prediction(baseline_key)
        # f_base and signal_shape are (1/σ)(dσ/dx) × scale_factor
        # Absolute: dσ/dx_i = f_i * σ / scale_factor ... but f_i is already
        # (1/σ)(dσ/dx)*scale_factor, so f_i/scale_factor * σ = dσ/dx
        # Actually simpler: combined normalised = (f_base*σ_base + μ*f_signal*σ_signal) / (σ_base + μ*σ_signal)
        # The scale_factor cancels in the ratio.
        num = f_base * sigma_baseline + mu * signal_shape * sigma_signal
        den = sigma_baseline + mu * sigma_signal
        return num / den

    def __repr__(self):
        return f"Observable(name={self.name!r}, nbins={self.nbins}, label={self.label!r})"
