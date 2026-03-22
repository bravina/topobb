"""
plotter.py — ATLAS-style plots for normalised differential cross sections.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from pathlib import Path
from observable import Observable

# Try atlasify first, then mplhep, then fallback
_STYLE = None
try:
    import atlasify
    _STYLE = "atlasify"
except ImportError:
    try:
        import mplhep as hep
        _STYLE = "mplhep"
    except ImportError:
        _STYLE = "fallback"


def _apply_atlas_style():
    """Apply ATLAS-like plot style."""
    if _STYLE == "mplhep":
        import mplhep as hep
        plt.style.use(hep.style.ATLAS)
    else:
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.linewidth": 1.3,
            "xtick.major.size": 8,
            "xtick.minor.size": 4,
            "ytick.major.size": 8,
            "ytick.minor.size": 4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        })


def _atlas_label(ax, label="TopoBB", status="Internal", lumi=140, energy=13):
    """Add ATLAS label to axes."""
    if _STYLE == "atlasify":
        import atlasify as atl
        atl.atlasify(
            ax, atlas=status,
            sub_text=label + f"\n$\\sqrt{{s}}={energy}$ TeV, {lumi} fb$^{{-1}}$",
        )
    elif _STYLE == "mplhep":
        import mplhep as hep
        hep.atlas.text(status, ax=ax, loc=1)
        ax.text(
            0.05, 0.78, label + f"\n$\\sqrt{{s}}={energy}$ TeV, {lumi} fb$^{{-1}}$",
            transform=ax.transAxes, fontsize=12, va="top",
        )
    else:
        ax.text(
            0.05, 0.95,
            r"$\bf{ATLAS}$ " + f"{status}\n{label}\n"
            f"$\\sqrt{{s}}={energy}$ TeV, {lumi} fb$^{{-1}}$",
            transform=ax.transAxes, fontsize=13, va="top",
        )


class Plotter:
    """
    Makes ATLAS-style plots of normalised differential cross sections.

    Parameters
    ----------
    output_dir : str
        Directory to save plots.
    experiment_label : str
        Experiment label shown on plot (e.g. "TopoBB").
    overflow_cap_factor : float
        The overflow bin is displayed with width = factor × second-to-last bin.
    """

    COLOURS = {
        "pwg_tt":       "#8B0000",
        "pwg_ttbb":     "#00008B",
        "pwg_tt_sig":   "#CC2222",
        "pwg_ttbb_sig": "#2222CC",
        "data_stat":    "#666666",
        "data_tot":     "#CCCCCC",
    }

    def __init__(
        self,
        output_dir: str = "plots",
        experiment_label: str = "TopoBB",
        overflow_cap_factor: float = 1.5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_label = experiment_label
        self.overflow_cap_factor = overflow_cap_factor

    def _display_edges(self, obs: Observable) -> np.ndarray:
        """
        Build bin edges for plotting, capping any overflow-like last bin
        (detected when its width is > 3× the second-to-last bin).
        """
        edges = []
        for lo, hi in obs.bins:
            edges.append(lo)
        last_lo, last_hi = obs.bins[-1]
        prev_width = obs.bins[-2][1] - obs.bins[-2][0]
        last_width = last_hi - last_lo if not np.isinf(last_hi) else np.inf

        if np.isinf(last_width) or last_width > 3 * prev_width:
            last_hi = last_lo + self.overflow_cap_factor * prev_width
        edges.append(last_hi)
        return np.array(edges, dtype=float)

    def plot_observable(
        self,
        obs: Observable,
        signal_shape: np.ndarray,
        sigma_baseline_tt: float,
        sigma_baseline_ttbb: float,
        sigma_signal: float,
        mu: float = 1.0,
        mu_label: str = "",
        filename: str = "",
        show: bool = False,
        y_range: tuple = None,
        ratio_range: tuple = (0.5, 1.5),
    ):
        """
        Plot normalised differential cross section with data and predictions.
        """
        _apply_atlas_style()

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(8, 7.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )

        # --- Bin geometry ---
        edges = self._display_edges(obs)
        centres = (edges[:-1] + edges[1:]) / 2.0
        widths = edges[1:] - edges[:-1]
        half_w = widths / 2.0

        # --- Data ---
        data = obs.data
        stat = obs.stat_errors
        tot = obs.total_error

        # --- Predictions ---
        pred_tt = obs.get_prediction("tt (5FS)")
        pred_ttbb = obs.get_prediction("ttbb (4FS)")

        mu_str = f"$\\mu={mu:.2f}$" if not mu_label else mu_label
        pred_tt_sig = obs.combined_prediction(
            "tt (5FS)", signal_shape, sigma_baseline_tt, sigma_signal, mu,
        )
        pred_ttbb_sig = obs.combined_prediction(
            "ttbb (4FS)", signal_shape, sigma_baseline_ttbb, sigma_signal, mu,
        )

        # =================================================================
        # TOP PANEL
        # =================================================================

        # Uncertainty bands (drawn as rectangles per bin)
        for i in range(obs.nbins):
            # Total band (light)
            ax_top.add_patch(Rectangle(
                (edges[i], data[i] - tot[i]), widths[i], 2 * tot[i],
                facecolor=self.COLOURS["data_tot"], edgecolor="none",
                zorder=1, label="Stat.+Syst." if i == 0 else None,
            ))
            # Stat band (dark)
            ax_top.add_patch(Rectangle(
                (edges[i], data[i] - stat[i]), widths[i], 2 * stat[i],
                facecolor=self.COLOURS["data_stat"], edgecolor="none",
                zorder=2, label="Stat." if i == 0 else None,
            ))

        # Data markers
        ax_top.errorbar(
            centres, data, xerr=half_w,
            fmt="ko", markersize=5, capsize=0, linewidth=1.2,
            label="Data", zorder=5,
        )

        # Prediction step lines
        self._draw_step(
            ax_top, edges, pred_tt,
            color=self.COLOURS["pwg_tt"], ls="-", lw=1.8,
            label=r"Powheg+Py8 $t\bar{t}$ (5FS)",
        )
        self._draw_step(
            ax_top, edges, pred_ttbb,
            color=self.COLOURS["pwg_ttbb"], ls="-", lw=1.8,
            label=r"Powheg+Py8 $t\bar{t}b\bar{b}$ (4FS)",
        )
        self._draw_step(
            ax_top, edges, pred_tt_sig,
            color=self.COLOURS["pwg_tt_sig"], ls="--", lw=1.8,
            label=rf"Pwg $t\bar{{t}}$ + MG sig. ({mu_str})",
        )
        self._draw_step(
            ax_top, edges, pred_ttbb_sig,
            color=self.COLOURS["pwg_ttbb_sig"], ls="--", lw=1.8,
            label=rf"Pwg $t\bar{{t}}b\bar{{b}}$ + MG sig. ({mu_str})",
        )

        # Y-axis label
        unit_str = obs.units
        ax_top.set_ylabel(
            r"$\frac{1000}{\sigma_{\mathrm{fid}}}"
            r"\;\frac{d\sigma_{\mathrm{fid}}}{d\,"
            + obs.label.strip("$")
            + r"}$  ["
            + unit_str
            + r"$^{-1}$]",
            fontsize=16,
        )
        if y_range:
            ax_top.set_ylim(y_range)
        else:
            ymax = max(np.max(data + tot), np.max(pred_tt), np.max(pred_ttbb))
            ax_top.set_ylim(0, ymax * 1.35)

        ax_top.tick_params(labelbottom=False)

        # Position legend to avoid overlap with ATLAS label (top-left)
        ax_top.legend(
            fontsize=8.5, ncol=2, loc="upper center",
            bbox_to_anchor=(0.58, 1.0), handlelength=2.2,
            columnspacing=1.0,
        )

        _atlas_label(ax_top, label=self.experiment_label)

        # Overflow indicator in top panel
        last_lo, last_hi = obs.bins[-1]
        prev_w = obs.bins[-2][1] - obs.bins[-2][0]
        is_overflow = np.isinf(last_hi) or (last_hi - last_lo) > 3 * prev_w
        if is_overflow:
            ax_top.annotate(
                r"$\geq$" + f"{last_lo:.0f}",
                xy=(edges[-1], 0), fontsize=9, ha="right", va="bottom",
                color="grey",
            )

        # =================================================================
        # BOTTOM PANEL (Pred. / Data)
        # =================================================================
        ratio_tot = tot / data
        ratio_stat = stat / data

        for i in range(obs.nbins):
            ax_bot.add_patch(Rectangle(
                (edges[i], 1 - ratio_tot[i]), widths[i], 2 * ratio_tot[i],
                facecolor=self.COLOURS["data_tot"], edgecolor="none", zorder=1,
            ))
            ax_bot.add_patch(Rectangle(
                (edges[i], 1 - ratio_stat[i]), widths[i], 2 * ratio_stat[i],
                facecolor=self.COLOURS["data_stat"], edgecolor="none", zorder=2,
            ))

        ax_bot.axhline(1.0, color="black", lw=0.8, ls="-", zorder=0)

        self._draw_step(ax_bot, edges, pred_tt / data,
                        color=self.COLOURS["pwg_tt"], ls="-", lw=1.8)
        self._draw_step(ax_bot, edges, pred_ttbb / data,
                        color=self.COLOURS["pwg_ttbb"], ls="-", lw=1.8)
        self._draw_step(ax_bot, edges, pred_tt_sig / data,
                        color=self.COLOURS["pwg_tt_sig"], ls="--", lw=1.8)
        self._draw_step(ax_bot, edges, pred_ttbb_sig / data,
                        color=self.COLOURS["pwg_ttbb_sig"], ls="--", lw=1.8)

        ax_bot.set_ylabel("Pred./Data")
        ax_bot.set_xlabel(f"{obs.label} [{obs.units}]", fontsize=16)
        ax_bot.set_ylim(ratio_range)
        ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

        # --- Save ---
        if not filename:
            filename = f"diff_xs_{obs.name}"
        for ext in ["pdf", "png"]:
            outpath = self.output_dir / f"{filename}.{ext}"
            fig.savefig(outpath, bbox_inches="tight", dpi=150)
        print(f"Plot saved: {self.output_dir / filename}.{{pdf,png}}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def _draw_step(ax, edges, values, **kwargs):
        """Draw a step histogram using horizontal lines per bin."""
        for i in range(len(values)):
            kw = {k: v for k, v in kwargs.items()}
            if i > 0:
                kw.pop("label", None)
            ax.plot([edges[i], edges[i + 1]], [values[i], values[i]], **kw)
            # Vertical connectors between bins
            if i > 0:
                kw_nolab = {k: v for k, v in kwargs.items() if k != "label"}
                ax.plot(
                    [edges[i], edges[i]], [values[i - 1], values[i]],
                    **kw_nolab,
                )
