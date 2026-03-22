# TopoBB Reinterpretation Framework

Fits a signal strength parameter μ for a new-physics signal on top of
SM baselines (Powheg+Pythia8 tt̄ 5FS or tt̄bb̄ 4FS), using unfolded
ATLAS HEPdata results from [ATLAS-2024-I2809112](https://www.hepdata.net/record/ins2809112).

## Structure

```
topobb/
├── mapping.py        # Central registry: observable names, regions, labels, file roots
├── loader.py         # Parse HEPdata YAML (fiducial, differential, correlation, bootstraps)
├── observable.py     # Observable class: bins, data, errors, predictions, covariance
├── fitter.py         # Fitter class: χ² minimisation for μ (fiducial + shapes)
├── plotter.py        # ATLAS-style two-panel differential cross-section plots
├── scan.py           # μ scan, profile likelihood, baseline comparison plots
├── signal.py         # Load signal from Rivet YODA, compute shape-only uncertainties
├── validation.py     # Sanity checks on inputs and covariance matrices
├── run.py            # Demo script (placeholder signal)
├── HEPdata_inputs/   # HEPdata YAML files (data, correlation matrices, bootstraps)
├── YODA_inputs/      # Rivet YODA output for the signal process
├── FitResults/       # Fit output text files (auto-created)
└── plots/            # Output plots as PDF + PNG (auto-created)
```

## Installation

```bash
uv sync          # install dependencies from uv.lock
uv run run.py    # run the demo with a placeholder signal
```

Dependencies: numpy, scipy, matplotlib, pyyaml.
Optional: mplhep or atlasify for ATLAS-style plot decorations.


## Observable naming and mapping

All observables are managed through `mapping.py`, which provides a central
registry of 71 observable/region combinations across 5 fiducial regions
(≥2j≥2b, ≥3j≥3b, ≥4j≥4b, ≥4j≥3b≥1l/c, ≥5j≥4b≥1l/c).

Each entry maps a human-readable short name + region to:
- a **HEPdata file root** (e.g. `HT_had_3j3b`) used to locate YAML files
- a **YODA histogram ID** (e.g. `d71-x01-y01`) for signal extraction
- a **LaTeX label** and **units** for plotting

```python
import mapping

info = mapping.get("HT_had", "3j3b")
info.hepdata_root   # "HT_had_3j3b"
info.yoda_id        # "d71-x01-y01"  (once filled in _YODA_IDS)
info.label          # r"$H_{\mathrm{T}}^{\mathrm{had}}$"
info.units          # "GeV"

mapping.list_observables()           # all 40 unique short names
mapping.get_all_in_region("3j3b")    # all 27 ObsInfo in ≥3b

# Load an Observable directly:
obs = mapping.load_observable("HT_had", "3j3b")
```

### HEPdata file naming

Each observable has three YAML files in `HEPdata_inputs/`:

| Suffix | Content |
|---|---|
| `{root}_stats.yaml` | Normalised differential XS: measured data with stat + syst errors, MC predictions |
| `{root}_corr_mtrx.yaml` | Correlation matrix across bins |
| `{root}_results.yaml` | Bootstrap replicas (1000 replicas × N bins) |

The fiducial cross sections live in `fid_xsec_systematics.yaml`.

### YODA ID mapping

The `_YODA_IDS` dict in `mapping.py` connects each (observable, region) to its
Rivet histogram identifier. Fill these in from your Rivet routine:

```python
_YODA_IDS = {
    ("HT_had", "3j3b"):  "d71-x01-y01",
    ("HT_had", "4j4b"):  "d72-x01-y01",
    # ...
}
```

Once populated, `mapping.get_signal_map("3j3b")` returns a
`{hepdata_root: yoda_id}` dict ready for the signal loader.


## Running a fit with the signal

### Step 1: Load the signal from YODA

Place your Rivet YODA output (run with weight variations) under `YODA_inputs/`.
Then use the `SignalLoader`:

```python
from signal import SignalLoader
import mapping

sig = SignalLoader(
    yoda_path="YODA_inputs/Rivet.yoda",
    analysis="ATLAS_2024_I2809112",
    observable_map=mapping.get_signal_map("3j3b"),  # all mapped observables
    fiducial_xsec=5.0,   # fb — set this to your signal's fiducial σ
)
pred = sig.load()
pred.summary()
```

The signal loader:
- Parses all YODA object types (HISTO1D, SCATTER2D, BinnedHisto, BinnedEstimate)
- Detects normalisation convention and converts to the ×1000 HEPdata format
- Computes **shape-only** uncertainties from weight variations:
  - **μR**: envelope of `MUR2_MUF1_PDF260400` and `MUR0.5_MUF1_PDF260400`
  - **μF**: envelope of `MUR1_MUF2_PDF260400` and `MUR1_MUF0.5_PDF260400`
  - **PDF**: RMS of 100 NNPDF3.0 replicas (`PDF260401`–`PDF260500`)

### Step 2: Load data and fit

```python
import mapping
from fitter import Fitter

# Load observables
obs_ht  = mapping.load_observable("HT_had", "3j3b")
obs_mbb = mapping.load_observable("Mass_b1b2", "3j3b")

# Fit μ with Powheg tt̄ (5FS) as baseline
fitter = Fitter(
    baseline_key_fid=r"$t\bar{t}$ (5FS)",
    baseline_key_diff="tt (5FS)",
    fiducial_region=mapping.get_fiducial_key("3j3b"),
    signal_sigma=pred.fiducial_xsec,
    signal_shapes=pred.get_signal_shapes_dict(),
)

# Fiducial + two differential shapes
result = fitter.fit(
    observables=[obs_ht, obs_mbb],
    include_fiducial=True,
    use_bootstraps=True,   # cross-observable stat correlations from bootstraps
)
print(result)
result.save("FitResults/fit_tt5FS_HT_mbb.txt")
```

### Step 3: Plot

```python
from plotter import Plotter
import loader

fid = loader.load_fiducial(region=mapping.get_fiducial_key("3j3b"))
sigma_tt   = fid["predictions"]["Powheg+Pythia 8 $t\\bar{t}$ (5FS)"]
sigma_ttbb = fid["predictions"]["Powheg+Pythia 8 $t\\bar{t}b\\bar{b}$ (4FS)"]

plotter = Plotter(output_dir="plots", experiment_label="TopoBB")
plotter.plot_observable(
    obs=obs_ht,
    signal_shape=pred.shapes[obs_ht.name],
    sigma_baseline_tt=sigma_tt,
    sigma_baseline_ttbb=sigma_ttbb,
    sigma_signal=pred.fiducial_xsec,
    mu=result.mu,
)
```

### Step 4: μ scan and baseline comparison

```python
from scan import MuScanner, compare_baselines

scanner = MuScanner(fitter, observables=[obs_ht], include_fiducial=True)
scan_result = scanner.scan(mu_range=(-2, 10))
scanner.plot_scan(scan_result, filename="mu_scan_HT")

compare_baselines(
    baselines={
        r"Pwg $t\bar{t}$ (5FS)":          (r"$t\bar{t}$ (5FS)",          "tt (5FS)"),
        r"Pwg $t\bar{t}b\bar{b}$ (4FS)":  (r"$t\bar{t}b\bar{b}$ (4FS)", "ttbb (4FS)"),
    },
    observables=[obs_ht],
    signal_sigma=pred.fiducial_xsec,
    signal_shapes=pred.get_signal_shapes_dict(),
    fiducial_region=mapping.get_fiducial_key("3j3b"),
)
```

### CLI for signal injection

```bash
uv run signal.py \
    --yoda YODA_inputs/Rivet.yoda \
    --fiducial-xsec 5.0 \
    --map HT_had_3j3b d71-x01-y01 \
    --map Mass_b1b2_3j3b d85-x01-y01 \
    --hepdata-dir HEPdata_inputs \
    --output-dir HEPdata_inputs_with_signal
```


## How the fit works

The χ² has two components:

**Fiducial:**  χ²_fid = (σ_meas − σ_base − μ·σ_sig)² / δσ²

**Shape** (per observable, last bin dropped for normalisation constraint):
χ²_shape = Δf^T · C⁻¹ · Δf, where Δf = f_meas − f_pred(μ) and

f_pred(μ) = (f_base·σ_base + μ·f_sig·σ_sig) / (σ_base + μ·σ_sig)

For multiple observables the combined covariance is built from:
- **Statistical correlations:** 1000 bootstrap replicas (consistent across observables)
- **Systematic correlations:** Σ_k δ_k δ_k^T (each named source fully correlated)

The μ uncertainty is the Δχ² = 1 interval.


## Validation

```bash
uv run validation.py
```

Checks: data/prediction normalisation, correlation matrix properties (symmetric,
positive semi-definite), covariance conditioning, bootstrap consistency with
reported stat errors, and combined cross-observable covariance health.


## Notes

- The HEPdata input files are provided by the ATLAS Collaboration under CC-BY-4.0.
- When combining multiple observables, always use `use_bootstraps=True` — the
  bootstrap replicas are the only source of cross-observable statistical correlations
  (they share the same Poisson weights across all observables).
- Overflow bins are detected and capped in plots (display only; fit data is unchanged).
- The baseline prediction keys differ between fiducial (LaTeX, e.g.
  `$t\bar{t}$ (5FS)`) and differential (plain text, e.g. `tt (5FS)`).
  The Fitter takes both via `baseline_key_fid` and `baseline_key_diff`.


## License

MIT