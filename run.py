#!/usr/bin/env python3
"""
run.py — Main script demonstrating the TopoBB reinterpretation framework.

Usage:
    python run.py
"""

import numpy as np
from observable import Observable
from fitter import Fitter
from plotter import Plotter
import loader
import mapping


# =========================================================================
# Prediction key constants
# =========================================================================
# Fiducial predictions use LaTeX labels; differential use plain text.
KEY_FID_TT   = r"$t\bar{t}$ (5FS)"
KEY_FID_TTBB = r"$t\bar{t}b\bar{b}$ (4FS)"
KEY_DIFF_TT   = "tt (5FS)"
KEY_DIFF_TTBB = "ttbb (4FS)"


def main():
    # =========================================================================
    # 1. Load fiducial results
    # =========================================================================
    REGION = "3j3b"
    fid_key = mapping.get_fiducial_key(REGION)

    print("=" * 70)
    print("FIDUCIAL CROSS SECTIONS")
    print("=" * 70)
    fid = loader.load_fiducial(region=fid_key)
    print(f"  Measured σ_fid = {fid['value']:.1f} ± {fid['stat']:.3f} (stat) fb")
    print(f"  Number of systematic sources: {len(fid['syst'])}")
    print("  MC predictions:")
    for k, v in fid["predictions"].items():
        print(f"    {k:50s} : {v:.1f} fb")

    # =========================================================================
    # 2. Load observable via mapping
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING OBSERVABLE: H_T^had (≥3b)")
    print("=" * 70)
    obs_ht = mapping.load_observable("HT_had", REGION)
    ht_root = mapping.get("HT_had", REGION).hepdata_root

    print(obs_ht)
    print(f"  Bins: {obs_ht.bins}")
    print(f"  Data: {obs_ht.data}")
    print(f"  Stat errors: {obs_ht.stat_errors}")
    print(f"  Syst sources: {len(obs_ht.syst_breakdown)}")
    print(f"  Bootstrap shape: {obs_ht.bootstraps.shape}")
    print(f"  Correlation matrix:\n{obs_ht.correlation}")

    # Validate: bootstrap stat errors vs reported stat errors
    bs_std = np.std(obs_ht.bootstraps * obs_ht.scale_factor, axis=0)
    print(f"\n  Bootstrap std (×1000): {bs_std}")
    print(f"  Reported stat errors:  {obs_ht.stat_errors}")

    # =========================================================================
    # 3. Define a PLACEHOLDER signal
    # =========================================================================
    # TODO: Replace with actual MadGraph signal via SignalLoader.
    print("\n" + "=" * 70)
    print("PLACEHOLDER SIGNAL (replace with MadGraph topobb)")
    print("=" * 70)

    sigma_signal = 5.0  # placeholder (fb)
    print(f"  Signal σ_fid = {sigma_signal:.1f} fb")

    signal_shape_ht = np.array([1.2, 2.8, 2.5, 1.8, 1.1, 0.12])
    bin_widths = obs_ht.bin_widths
    integral = np.sum(signal_shape_ht * bin_widths) / obs_ht.scale_factor
    print(f"  Signal shape integral check: {integral:.4f} (should be ~1.0)")
    signal_shape_ht = signal_shape_ht / integral
    integral2 = np.sum(signal_shape_ht * bin_widths) / obs_ht.scale_factor
    print(f"  After renorm: {integral2:.4f}")
    print(f"  Signal shape: {signal_shape_ht}")

    # Signal shapes dict keyed by hepdata_root (as expected by Fitter)
    signal_shapes = {ht_root: signal_shape_ht}

    # =========================================================================
    # 4. Fit μ
    # =========================================================================
    print("\n" + "=" * 70)
    print("FITTING: Powheg tt (5FS) + μ × signal")
    print("=" * 70)

    fitter_tt = Fitter(
        baseline_key_fid=KEY_FID_TT,
        baseline_key_diff=KEY_DIFF_TT,
        fiducial_region=fid_key,
        signal_sigma=sigma_signal,
        signal_shapes=signal_shapes,
    )

    # Fiducial only
    res_fid_only = fitter_tt.fit(observables=[], include_fiducial=True)
    print("\n--- Fiducial only ---")
    print(res_fid_only)
    res_fid_only.save("FitResults/fit_tt5FS_fiducial_only.txt")

    # Fiducial + H_T^had
    res_fid_ht = fitter_tt.fit(
        observables=[obs_ht], include_fiducial=True, use_bootstraps=False
    )
    print("\n--- Fiducial + H_T^had ---")
    print(res_fid_ht)
    res_fid_ht.save("FitResults/fit_tt5FS_fiducial_HTHad.txt")

    # Shape only (no fiducial)
    res_shape_only = fitter_tt.fit(
        observables=[obs_ht], include_fiducial=False, use_bootstraps=False
    )
    print("\n--- H_T^had shape only (no fiducial) ---")
    print(res_shape_only)
    res_shape_only.save("FitResults/fit_tt5FS_HTHad_shape_only.txt")

    # Goodness of fit at μ=0 (baseline only)
    gof_mu0 = fitter_tt.goodness_of_fit(
        observables=[obs_ht], mu=0.0, include_fiducial=True,
        label="Powheg tt (5FS), μ=0",
    )
    print("\n--- Goodness of fit at μ=0 ---")
    print(gof_mu0)
    gof_mu0.save("FitResults/gof_tt5FS_mu0.txt")

    # --- Fit with Powheg ttbb (4FS) baseline ---
    print("\n" + "=" * 70)
    print("FITTING: Powheg ttbb (4FS) + μ × signal")
    print("=" * 70)

    fitter_ttbb = Fitter(
        baseline_key_fid=KEY_FID_TTBB,
        baseline_key_diff=KEY_DIFF_TTBB,
        fiducial_region=fid_key,
        signal_sigma=sigma_signal,
        signal_shapes=signal_shapes,
    )

    res_ttbb = fitter_ttbb.fit(
        observables=[obs_ht], include_fiducial=True, use_bootstraps=False
    )
    print("\n--- Fiducial + H_T^had (ttbb 4FS baseline) ---")
    print(res_ttbb)
    res_ttbb.save("FitResults/fit_ttbb4FS_fiducial_HTHad.txt")

    # =========================================================================
    # 5. Plots
    # =========================================================================
    print("\n" + "=" * 70)
    print("MAKING PLOTS")
    print("=" * 70)

    plotter = Plotter(output_dir="plots", experiment_label="TopoBB")

    sigma_tt = fid["predictions"][
        [k for k in fid["predictions"] if KEY_FID_TT in k and "Powheg" in k][0]
    ]
    sigma_ttbb = fid["predictions"][
        [k for k in fid["predictions"] if KEY_FID_TTBB in k and "Powheg+Pythia" in k][0]
    ]
    print(f"  σ_tt (5FS)   = {sigma_tt:.1f} fb")
    print(f"  σ_ttbb (4FS) = {sigma_ttbb:.1f} fb")

    plotter.plot_observable(
        obs=obs_ht,
        signal_shape=signal_shape_ht,
        sigma_baseline_tt=sigma_tt,
        sigma_baseline_ttbb=sigma_ttbb,
        sigma_signal=sigma_signal,
        mu=1.0,
        mu_label=r"$\mu=1$",
        filename="HTHad_3b_mu1",
    )

    plotter.plot_observable(
        obs=obs_ht,
        signal_shape=signal_shape_ht,
        sigma_baseline_tt=sigma_tt,
        sigma_baseline_ttbb=sigma_ttbb,
        sigma_signal=sigma_signal,
        mu=res_fid_ht.mu,
        mu_label=f"$\\mu={res_fid_ht.mu:.2f}$",
        filename="HTHad_3b_mu_bestfit",
    )

    # =========================================================================
    # 6. Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    from validation import validate_observable, validate_fiducial

    validate_fiducial()
    validate_observable(obs_ht)

    # =========================================================================
    # 7. μ scan and profile likelihood
    # =========================================================================
    print("\n" + "=" * 70)
    print("μ SCAN")
    print("=" * 70)
    from scan import MuScanner, compare_baselines

    scanner_tt = MuScanner(
        fitter_tt, observables=[obs_ht],
        use_bootstraps=False, include_fiducial=True,
    )
    scan_tt = scanner_tt.scan(mu_range=(-2.0, 10.0))
    scanner_tt.plot_scan(
        scan_tt,
        title=r"Powheg+Py8 $t\bar{t}$ (5FS) + signal",
        filename="mu_scan_tt5FS",
    )

    compare_baselines(
        baselines={
            r"Pwg $t\bar{t}$ (5FS)": (KEY_FID_TT, KEY_DIFF_TT),
            r"Pwg $t\bar{t}b\bar{b}$ (4FS)": (KEY_FID_TTBB, KEY_DIFF_TTBB),
        },
        observables=[obs_ht],
        signal_sigma=sigma_signal,
        signal_shapes=signal_shapes,
        fiducial_region=fid_key,
        mu_range=(-2.0, 10.0),
        output_dir="plots",
        filename="mu_scan_comparison",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
