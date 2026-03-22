# TopoBB Reinterpretation Framework

Fits a signal strength parameter μ for a new-physics signal (MadGraph topobb)
added on top of SM baselines (Powheg tt 5FS or Powheg ttbb 4FS), using
unfolded ATLAS HEPdata results.

## Structure

```
topobb_framework/
├── loader.py         # Parse HEPdata YAML (fiducial, differential, correlation, bootstraps)
├── observable.py     # Observable class: bins, data, errors, predictions, covariance
├── fitter.py         # Fitter class: χ² minimisation for μ (fiducial + shapes)
├── plotter.py        # ATLAS-style two-panel differential cross-section plots
├── scan.py           # μ scan, profile likelihood, baseline comparison plots
├── signal.py         # Load signal from Rivet YODA, compute shape-only uncertainties
├── validation.py     # Sanity checks on inputs and covariance matrices
├── run.py            # Demo script (placeholder signal)
├── HEPdata_inputs/   # Input YAML files
├── FitResults/       # Fit output text files
└── plots/            # Output plots (PDF + PNG)
```

## Quick Start

```bash
# Run the full demo (placeholder signal)
python run.py

# Run validation only
python validation.py
```

## Plugging In Your MadGraph Signal

After running Rivet with the analysis routine and weight variations:

```python
from signal import SignalLoader

sig = SignalLoader(
    yoda_path="path/to/Rivet.yoda",
    analysis="ATLAS_2024_I2809112",
    observable_map={
        "H__T___had__in__3b": "d71-x01-y01",
        # add more observables as needed
    },
    fiducial_xsec=5.0,  # in fb (or use fiducial_histo_id to extract from YODA)
)
pred = sig.load()
pred.summary()

# Use directly with the framework
fitter = Fitter(
    baseline_key_fid=...,
    baseline_key_diff=...,
    signal_sigma=pred.fiducial_xsec,
    signal_shapes=pred.get_signal_shapes_dict(),
)
```

The signal loader automatically:
- Parses the YODA file (supports YODA_HISTO1D, YODA_SCATTER2D, BinnedHisto, BinnedEstimate)
- Detects normalisation convention and converts to the ×1000 HEPdata format
- Computes **shape-only** uncertainties from weight variations:
  - **muR**: envelope of MUR2_MUF1_PDF260400 and MUR0.5_MUF1_PDF260400 (independent muR ×2/×0.5)
  - **muF**: envelope of MUR1_MUF2_PDF260400 and MUR1_MUF0.5_PDF260400 (independent muF ×2/×0.5)
  - **PDF**: RMS of 100 NNPDF3.0 replicas (PDF260401–260500, nominal = PDF260400)

CLI usage:
```bash
python signal.py \
    --yoda Rivet.yoda \
    --fiducial-xsec 5.0 \
    --map H__T___had__in__3b d71-x01-y01 \
    --hepdata-dir HEPdata_inputs \
    --output-dir HEPdata_inputs_with_signal
```

## Adding More Observables

1. Download 3 more HEPdata YAML files per observable (Diff_XS, Corr_mtrx, Bootstrap)
   and place them in `HEPdata_inputs/`.

2. Load them:
```python
obs_new = Observable.from_hepdata(
    name="<file_identifier>",     # e.g. "m_bb__in__3b"
    label=r"$m_{bb}$",
    units="GeV",
)
```

3. Provide the signal shape for this observable:
```python
signal_shapes["<name>"] = np.array([...])
```

4. Fit multiple observables together:
```python
result = fitter.fit(
    observables=[obs_ht, obs_new],
    use_bootstraps=True,    # IMPORTANT for cross-observable correlations
    include_fiducial=True,
)
```

## How the Fit Works

The χ² has two components:

**Fiducial:**
```
χ²_fid = (σ_meas - σ_base - μ·σ_sig)² / δσ²
```

**Shape (per observable, last bin dropped):**
```
χ²_shape = Δf^T · C⁻¹ · Δf
```
where `Δf = f_meas - f_pred(μ)` and `f_pred(μ)` is the re-normalised
combined prediction:
```
f_pred = (f_base·σ_base + μ·f_sig·σ_sig) / (σ_base + μ·σ_sig)
```

For multiple observables, the combined covariance is:
- **Statistical:** from the 1000 bootstraps (cross-observable correlations)
- **Systematic:** Σ_k δ_k δ_k^T (each source 100% correlated across observables)

## Dependencies

- Python 3.8+
- numpy, scipy, matplotlib, pyyaml
- Optional: mplhep or atlasify (for ATLAS-style plots; works without them)

## Key Notes

- The correlation matrix from HEPdata encodes total (stat+syst) correlations
  within a single observable. For combining observables, we build the full
  cross-observable covariance from bootstraps (stat) + syst breakdown.
- Bootstrap replicas are consistent across observables (same Poisson weights),
  which is what enables the cross-observable stat covariance.
- The last bin is always dropped from normalised distributions in the χ²
  (normalisation constraint makes the covariance singular over all N bins).
- Overflow bins are detected and capped in plots (display only; data is unchanged).
- The μ uncertainty is the Δχ² = 1 interval (≈ 1σ Gaussian equivalent).
