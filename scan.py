"""
scan.py — μ-scan and profile likelihood utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.stats import chi2 as chi2_dist
from fitter import Fitter, FitResult
from observable import Observable
from typing import Optional


class MuScanner:
    """
    Scan χ²(μ) and produce profile likelihood plots.

    Parameters
    ----------
    fitter : Fitter
        Configured fitter instance.
    observables : list of Observable
        Observables to include in the fit.
    use_bootstraps : bool
        Use bootstraps for cross-observable stat correlations.
    include_fiducial : bool
        Include fiducial χ² term.
    """

    def __init__(
        self,
        fitter: Fitter,
        observables: Optional[list] = None,
        use_bootstraps: bool = False,
        include_fiducial: bool = True,
    ):
        self.fitter = fitter
        self.observables = observables or []
        self.use_bootstraps = use_bootstraps
        self.include_fiducial = include_fiducial

    def scan(self, mu_range: tuple = (-2.0, 10.0), n_points: int = 200) -> dict:
        """
        Scan χ²(μ) over a range.

        Returns
        -------
        dict with keys:
            'mu'        : np.ndarray of μ values
            'chi2'      : np.ndarray of total χ²
            'chi2_fid'  : np.ndarray of fiducial χ²
            'chi2_shape': np.ndarray of shape χ²
            'delta_chi2': np.ndarray of Δχ² = χ² - χ²_min
            'mu_best'   : float
            'chi2_min'  : float
            'mu_1sigma' : (float, float) — μ ± 1σ interval
            'mu_2sigma' : (float, float) — μ ± 2σ interval
        """
        mu_arr = np.linspace(mu_range[0], mu_range[1], n_points)
        chi2_total = np.zeros(n_points)
        chi2_fid = np.zeros(n_points)
        chi2_shape = np.zeros(n_points)

        for i, mu in enumerate(mu_arr):
            if self.include_fiducial:
                chi2_fid[i] = self.fitter.chi2_fiducial(mu)
            chi2_shape[i] = self.fitter.chi2_shape(
                mu, self.observables, self.use_bootstraps
            )
            chi2_total[i] = chi2_fid[i] + chi2_shape[i]

        # Find minimum
        idx_min = np.argmin(chi2_total)
        mu_best = mu_arr[idx_min]
        chi2_min = chi2_total[idx_min]
        delta_chi2 = chi2_total - chi2_min

        # Find 1σ and 2σ intervals by interpolation
        mu_1sigma = self._find_interval(mu_arr, delta_chi2, 1.0)
        mu_2sigma = self._find_interval(mu_arr, delta_chi2, 4.0)

        return {
            "mu": mu_arr,
            "chi2": chi2_total,
            "chi2_fid": chi2_fid,
            "chi2_shape": chi2_shape,
            "delta_chi2": delta_chi2,
            "mu_best": mu_best,
            "chi2_min": chi2_min,
            "mu_1sigma": mu_1sigma,
            "mu_2sigma": mu_2sigma,
        }

    def plot_scan(
        self,
        scan_result: dict,
        title: str = "",
        filename: str = "mu_scan",
        output_dir: str = "plots",
        show: bool = False,
        show_components: bool = True,
    ):
        """
        Plot the Δχ²(μ) profile.

        Parameters
        ----------
        scan_result : dict from self.scan()
        show_components : bool
            Show individual χ²_fid and χ²_shape contributions.
        """
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.linewidth": 1.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        })

        mu = scan_result["mu"]
        dchi2 = scan_result["delta_chi2"]
        mu_best = scan_result["mu_best"]
        chi2_min = scan_result["chi2_min"]
        mu_1s = scan_result["mu_1sigma"]
        mu_2s = scan_result["mu_2sigma"]

        fig, ax = plt.subplots(figsize=(7, 5))

        # Main Δχ² curve
        ax.plot(mu, dchi2, "k-", lw=2, label=r"$\Delta\chi^2(\mu)$")

        # Components (offset to same minimum)
        if show_components:
            chi2_fid = scan_result["chi2_fid"]
            chi2_shape = scan_result["chi2_shape"]
            dchi2_fid = chi2_fid - chi2_fid[np.argmin(scan_result["chi2"])]
            dchi2_shape = chi2_shape - chi2_shape[np.argmin(scan_result["chi2"])]
            if self.include_fiducial and np.any(chi2_fid > 0):
                ax.plot(mu, chi2_fid, "--", color="#8B0000", lw=1.2,
                        label=r"$\chi^2_{\mathrm{fid}}$")
            if len(self.observables) > 0:
                ax.plot(mu, chi2_shape, "--", color="#00008B", lw=1.2,
                        label=r"$\chi^2_{\mathrm{shape}}$")

        # Horizontal lines for 1σ, 2σ
        ax.axhline(1.0, color="grey", ls=":", lw=1, alpha=0.7)
        ax.axhline(4.0, color="grey", ls=":", lw=1, alpha=0.7)
        ax.text(mu[-1], 1.0, r" $1\sigma$", va="bottom", fontsize=11, color="grey")
        ax.text(mu[-1], 4.0, r" $2\sigma$", va="bottom", fontsize=11, color="grey")

        # 1σ band
        if mu_1s[0] is not None and mu_1s[1] is not None:
            ax.axvspan(mu_1s[0], mu_1s[1], alpha=0.15, color="green",
                       label=rf"$\mu = {mu_best:.2f}^{{+{mu_1s[1]-mu_best:.2f}}}_{{-{mu_best-mu_1s[0]:.2f}}}$")

        # 2σ band
        if mu_2s[0] is not None and mu_2s[1] is not None:
            ax.axvspan(mu_2s[0], mu_2s[1], alpha=0.08, color="orange")

        # Best fit marker
        ax.axvline(mu_best, color="red", ls="-", lw=1, alpha=0.5)

        ax.set_xlabel(r"$\mu$", fontsize=16)
        ax.set_ylabel(r"$\Delta\chi^2$", fontsize=16)
        ax.set_ylim(0, min(10, np.max(dchi2)))
        ax.legend(fontsize=11, loc="upper left")

        if title:
            ax.set_title(title, fontsize=14)

        # Annotate
        ndf_shape = sum(obs.nbins - 1 for obs in self.observables)
        ndf = ndf_shape + (1 if self.include_fiducial else 0) - 1
        ax.text(
            0.97, 0.95,
            f"$\\chi^2_{{\\min}}/\\mathrm{{ndf}} = {chi2_min:.2f}/{ndf}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
        )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for ext in ["pdf", "png"]:
            fig.savefig(Path(output_dir) / f"{filename}.{ext}",
                        bbox_inches="tight", dpi=150)
        print(f"Scan plot saved: {output_dir}/{filename}.{{pdf,png}}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _find_interval(mu_arr, delta_chi2, level):
        """Find the μ interval where Δχ² < level by interpolation."""
        below = delta_chi2 < level
        if not np.any(below):
            return (None, None)

        indices = np.where(below)[0]
        # Interpolate crossings
        lo = mu_arr[indices[0]]
        hi = mu_arr[indices[-1]]

        # Refine by linear interpolation at crossings
        if indices[0] > 0:
            i = indices[0]
            f = (level - delta_chi2[i - 1]) / (delta_chi2[i] - delta_chi2[i - 1])
            lo = mu_arr[i - 1] + f * (mu_arr[i] - mu_arr[i - 1])
        if indices[-1] < len(mu_arr) - 1:
            i = indices[-1]
            f = (level - delta_chi2[i]) / (delta_chi2[i + 1] - delta_chi2[i])
            hi = mu_arr[i] + f * (mu_arr[i + 1] - mu_arr[i])

        return (lo, hi)


def compare_baselines(
    baselines: dict,
    observables: list,
    signal_sigma: float,
    signal_shapes: dict,
    fiducial_region: str = "$\\geq3b$",
    mu_range: tuple = (-2.0, 10.0),
    output_dir: str = "plots",
    filename: str = "baseline_comparison",
    use_bootstraps: bool = False,
):
    """
    Compare μ scans for different baselines on the same plot.

    Parameters
    ----------
    baselines : dict
        {label: (baseline_key_fid, baseline_key_diff)}
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.linewidth": 1.3,
        "legend.frameon": False,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ["#8B0000", "#00008B", "#006400", "#8B4513"]

    for idx, (label, (key_fid, key_diff)) in enumerate(baselines.items()):
        fitter = Fitter(
            baseline_key_fid=key_fid,
            baseline_key_diff=key_diff,
            fiducial_region=fiducial_region,
            signal_sigma=signal_sigma,
            signal_shapes=signal_shapes,
        )
        scanner = MuScanner(
            fitter, observables, use_bootstraps=use_bootstraps, include_fiducial=True,
        )
        result = scanner.scan(mu_range=mu_range)

        colour = colours[idx % len(colours)]
        mu_best = result["mu_best"]
        mu_1s = result["mu_1sigma"]
        err_up = (mu_1s[1] - mu_best) if mu_1s[1] else 0
        err_dn = (mu_best - mu_1s[0]) if mu_1s[0] else 0

        ax.plot(
            result["mu"], result["delta_chi2"],
            color=colour, lw=2,
            label=f"{label}: $\\mu={mu_best:.2f}^{{+{err_up:.2f}}}_{{-{err_dn:.2f}}}$",
        )

    ax.axhline(1.0, color="grey", ls=":", lw=1, alpha=0.7)
    ax.axhline(4.0, color="grey", ls=":", lw=1, alpha=0.7)
    ax.text(mu_range[1], 1.0, r" $1\sigma$", va="bottom", fontsize=11, color="grey")
    ax.text(mu_range[1], 4.0, r" $2\sigma$", va="bottom", fontsize=11, color="grey")

    ax.set_xlabel(r"$\mu$", fontsize=16)
    ax.set_ylabel(r"$\Delta\chi^2$", fontsize=16)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=10)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(Path(output_dir) / f"{filename}.{ext}",
                    bbox_inches="tight", dpi=150)
    print(f"Comparison plot saved: {output_dir}/{filename}.{{pdf,png}}")
    plt.close(fig)
    return fig
