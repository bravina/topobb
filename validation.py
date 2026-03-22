"""
validation.py — Sanity checks for Observable and Fitter inputs.

Run standalone: python validation.py
"""

import numpy as np
from observable import Observable
import loader


def validate_observable(obs: Observable, verbose: bool = True) -> dict:
    """
    Run a suite of sanity checks on an Observable.

    Returns dict of {check_name: (passed: bool, message: str)}.
    """
    results = {}

    # 1. Normalisation: data should integrate to ~1
    bw = obs.bin_widths
    integral = np.sum(obs.data * bw) / obs.scale_factor
    ok = abs(integral - 1.0) < 0.05
    results["data_normalisation"] = (ok, f"∫data = {integral:.4f} (expect ~1.0)")

    # 2. Prediction normalisation
    for key, pred in obs.predictions.items():
        integ = np.sum(pred * bw) / obs.scale_factor
        ok_p = abs(integ - 1.0) < 0.05
        results[f"pred_norm_{key}"] = (ok_p, f"∫{key} = {integ:.4f}")

    # 3. Correlation matrix: diagonal should be 1, symmetric, eigenvalues > 0
    if obs.correlation is not None:
        diag_ok = np.allclose(np.diag(obs.correlation), 1.0)
        sym_ok = np.allclose(obs.correlation, obs.correlation.T)
        eigvals = np.linalg.eigvalsh(obs.correlation)
        psd_ok = np.all(eigvals > -1e-10)
        results["corr_diagonal"] = (diag_ok, f"diag = {np.diag(obs.correlation)}")
        results["corr_symmetric"] = (sym_ok, "symmetric" if sym_ok else "NOT symmetric")
        results["corr_psd"] = (psd_ok, f"min eigenvalue = {eigvals[0]:.6e}")

    # 4. Covariance: positive definite (full and after dropping last bin)
    try:
        cov = obs.covariance
        eigvals_cov = np.linalg.eigvalsh(cov)
        results["cov_psd"] = (
            np.all(eigvals_cov > -1e-14),
            f"min eigval = {eigvals_cov[0]:.6e}",
        )
        n = obs.nbins - 1
        cov_sub = cov[:n, :n]
        eigvals_sub = np.linalg.eigvalsh(cov_sub)
        results["cov_sub_psd"] = (
            np.all(eigvals_sub > 0),
            f"min eigval (N-1) = {eigvals_sub[0]:.6e}",
        )
        cond = np.linalg.cond(cov_sub)
        results["cov_sub_condition"] = (
            cond < 1e8,
            f"condition number = {cond:.1f}",
        )
    except Exception as e:
        results["cov_computation"] = (False, str(e))

    # 5. Bootstrap consistency: bootstrap std vs reported stat errors
    if obs.bootstraps is not None:
        bs = obs.bootstraps * obs.scale_factor
        bs_std = np.std(bs, axis=0)
        ratio = bs_std / obs.stat_errors
        # Should be within factor ~2 (overflow bins can differ more)
        ratio_ok = np.all((ratio > 0.3) & (ratio < 5.0))
        results["bootstrap_stat_consistency"] = (
            ratio_ok,
            f"bootstrap_std/stat_err ratio per bin: {np.round(ratio, 3)}",
        )
        # Bootstrap means vs data
        bs_mean = np.mean(bs, axis=0)
        mean_ratio = bs_mean / obs.data
        mean_ok = np.all(np.abs(mean_ratio - 1.0) < 0.05)
        results["bootstrap_mean_consistency"] = (
            mean_ok,
            f"bootstrap_mean/data ratio: {np.round(mean_ratio, 4)}",
        )

    # 6. Systematic covariance positive semi-definite
    syst_cov = obs.syst_covariance
    eigvals_syst = np.linalg.eigvalsh(syst_cov)
    results["syst_cov_psd"] = (
        np.all(eigvals_syst > -1e-14),
        f"min eigval = {eigvals_syst[0]:.6e}",
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION: {obs.name} ({obs.nbins} bins)")
        print(f"{'='*60}")
        for name, (passed, msg) in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: {msg}")
        n_pass = sum(1 for ok, _ in results.values() if ok)
        n_total = len(results)
        print(f"\n  {n_pass}/{n_total} checks passed")

    return results


def validate_fiducial(region: str = "$\\geq3b$", verbose: bool = True) -> dict:
    """Validate fiducial cross section data."""
    results = {}
    fid = loader.load_fiducial(region=region)

    # Check measured value is positive and reasonable
    results["value_positive"] = (fid["value"] > 0, f"σ = {fid['value']:.1f} fb")

    # Check stat error is reasonable fraction
    frac = fid["stat"] / fid["value"]
    results["stat_frac"] = (frac < 0.2, f"stat/σ = {frac:.4f}")

    # Check syst errors have consistent signs (down should be ≤ 0 or up ≥ 0 for
    # most, but one-sided systematics break this — just check for obviously wrong)
    n_suspicious = 0
    for source, (down, up) in fid["syst"].items():
        # Both positive or both negative is fine (one-sided)
        # But if |down| or |up| > σ, that's suspicious
        if abs(down) > fid["value"] or abs(up) > fid["value"]:
            n_suspicious += 1
    results["syst_magnitudes"] = (
        n_suspicious == 0,
        f"{n_suspicious} sources with |shift| > σ_fid",
    )

    # Total uncertainty
    total_sq = fid["stat"] ** 2
    for _, (d, u) in fid["syst"].items():
        total_sq += ((abs(u) + abs(d)) / 2.0) ** 2
    total = np.sqrt(total_sq)
    results["total_uncertainty"] = (True, f"δσ_total = {total:.2f} fb ({100*total/fid['value']:.1f}%)")

    # Predictions in reasonable range
    for key, val in fid["predictions"].items():
        ratio = val / fid["value"]
        ok = 0.3 < ratio < 3.0
        results[f"pred_range_{key[:30]}"] = (ok, f"{val:.0f} fb (ratio={ratio:.2f})")

    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION: Fiducial ({region})")
        print(f"{'='*60}")
        for name, (passed, msg) in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: {msg}")
        n_pass = sum(1 for ok, _ in results.values() if ok)
        print(f"\n  {n_pass}/{len(results)} checks passed")

    return results


def validate_combined_covariance(
    observables: list, use_bootstraps: bool = True, verbose: bool = True
) -> dict:
    """
    Validate the combined covariance matrix across multiple observables.
    """
    from fitter import Fitter

    results = {}

    # Build the combined covariance (reusing Fitter's method)
    fitter = Fitter.__new__(Fitter)  # just borrow the method
    fitter.signal_shapes = {}

    sizes = [obs.nbins - 1 for obs in observables]
    total = sum(sizes)

    # Stat covariance from bootstraps
    if use_bootstraps and all(obs.bootstraps is not None for obs in observables):
        bs_list = []
        for obs in observables:
            bs = obs.bootstraps * obs.scale_factor
            bs_list.append(bs[:, : obs.nbins - 1])
        bs_all = np.hstack(bs_list)
        cov_stat = np.cov(bs_all, rowvar=False)
    else:
        cov_stat = np.zeros((total, total))
        offset = 0
        for obs, sz in zip(observables, sizes):
            sc = obs.stat_covariance[:sz, :sz]
            cov_stat[offset:offset + sz, offset:offset + sz] = sc
            offset += sz

    # Syst covariance
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

    cov_total = cov_stat + cov_syst

    # Checks
    eigvals = np.linalg.eigvalsh(cov_total)
    results["combined_psd"] = (
        np.all(eigvals > -1e-14),
        f"min eigval = {eigvals[0]:.6e}",
    )
    cond = np.linalg.cond(cov_total)
    results["combined_condition"] = (
        cond < 1e8,
        f"condition number = {cond:.1f}",
    )
    results["combined_size"] = (True, f"{total}×{total} ({' + '.join(str(s) for s in sizes)})")

    # Cross-observable correlation: extract off-diagonal blocks
    if len(observables) > 1:
        offset_i = 0
        for i, obs_i in enumerate(observables):
            offset_j = offset_i + sizes[i]
            for j, obs_j in enumerate(observables[i + 1:], start=i + 1):
                block = cov_total[offset_i:offset_i + sizes[i],
                                  offset_j:offset_j + sizes[j]]
                # Normalise to correlations
                diag_i = np.sqrt(np.diag(cov_total)[offset_i:offset_i + sizes[i]])
                diag_j = np.sqrt(np.diag(cov_total)[offset_j:offset_j + sizes[j]])
                corr_block = block / np.outer(diag_i, diag_j)
                max_corr = np.max(np.abs(corr_block))
                results[f"cross_corr_{obs_i.name}_{obs_j.name}"] = (
                    True,
                    f"max |ρ| = {max_corr:.3f}",
                )
                offset_j += sizes[j]
            offset_i += sizes[i]

    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION: Combined covariance ({len(observables)} observables)")
        print(f"{'='*60}")
        for name, (passed, msg) in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: {msg}")
        n_pass = sum(1 for ok, _ in results.values() if ok)
        print(f"\n  {n_pass}/{len(results)} checks passed")

    return results


if __name__ == "__main__":
    # Run all validations on the available data
    validate_fiducial()

    obs_ht = Observable.from_hepdata(
        "H__T___had__in__3b",
        r"$H_{\mathrm{T}}^{\mathrm{had}}$",
        "GeV",
    )
    validate_observable(obs_ht)

    # Single-observable combined covariance check
    validate_combined_covariance([obs_ht], use_bootstraps=True)
