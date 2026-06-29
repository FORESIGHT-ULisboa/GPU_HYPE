# GPU_HYPE

**`GPU_HYPE`** calibrates the [HYPE](https://hypeweb.smhi.se/) hydrological model
with a GPU-assisted, multi-objective evolutionary optimiser and turns the
calibrated parameter population into a **probabilistic (ensemble) streamflow
forecast**. It is the research code published alongside the paper, by
[FORESIGHT — Forecasting and Optimization for Resilient Environmental Systems through Investigation with Groundbreaking Hydrological Tools](https://foresight.tecnico.ulisboa.pt/).

Instead of returning a single "best" parameter set, the optimiser keeps a whole
population on a trade-off front between **reliability** (non-exceedance) and a
**streamflow error metric**, and aggregates the resulting simulations into
predictive uncertainty bands.

> **New here? Start with the notebooks** in [`notebooks/`](notebooks/), each a
> self-contained walkthrough on the bundled Türkheim catchment:
> [`00_setup_and_data`](notebooks/00_setup_and_data.ipynb) (environment, OpenCL
> check, data formats) ·
> [`01_calibration`](notebooks/01_calibration.ipynb) (the calibration tool) ·
> [`02_training_and_predicting`](notebooks/02_training_and_predicting.ipynb)
> (load the trained model, forecast an ensemble) ·
> [`03_results`](notebooks/03_results.ipynb) (hydrographs, reliability, skill).
> Run them under the project kernel (see [Installation](#installation)) with the
> working directory at the repo root.

---

## Features

| Category | What it provides |
|---|---|
| **Calibration** | `GPU_PSO` — multi-objective Particle-Swarm optimisation of HYPE parameters |
| **Models** | `HYPE` (runs the HYPE executable in parallel) and `ANN` (OpenCL regressor) |
| **Error models** | `Error` (OpenCL MAE/MSE kernels) and `NewErrorModel` (NumPy MAE/MSE/NSE) |
| **Probabilistic output** | population → predictive uncertainty bands (default 16 levels) |
| **Multi-objective** | reliability (non-exceedance) vs error, with NSGA-II sorting + crowding |
| **Visualisation** | `Display` / `PlotGPU` — Pareto front, PIT/Q-Q reliability, hydrograph bands |
| **Persistence** | `save` / `load` the whole optimiser as a pickle |
| **Diagnostics** *(optional)* | `functions2` hydrographs and skill metrics via the companion `forecast_performance` package |

---

## Installation

### 1 · Create a conda environment

```bat
conda create -n gpu_hype python=3.8
conda activate gpu_hype
```

### 2 · Install the package and dependencies

From the repository root:

```bat
pip install -e ".[dev]"
```

The advanced diagnostics in `functions2.py` (used by the optional section of the
results notebook) additionally need the companion `forecast_performance` package:

```bat
pip install -e ".[metrics]"
```

### 3 · Register the Jupyter kernel

```bat
python -m ipykernel install --user --name gpu_hype --display-name "gpu_hype"
```

### 4 · Open the notebooks

Open them in VS Code and select the **gpu_hype** kernel in the top-right kernel
picker, or run:

```bat
jupyter lab notebooks\
```

---

## System requirements

The runnable end-to-end pipeline targets **Windows with an OpenCL platform**:

- **Windows.** The HYPE model is invoked as `HYPEwithoutPopup4All.exe`
  (bundled in each HYPE folder) through `subprocess`.
- **OpenCL.** Importing `gpu_pso` builds an OpenCL context, so a working OpenCL
  platform must be available — a GPU, or a CPU OpenCL runtime. Check it with:
  ```python
  import pyopencl as cl
  print(cl.get_platforms())
  ```

---

## Core concepts

- **HYPE folder.** A standard HYPE setup directory (`info.txt`, `par.txt`,
  `GeoData.txt`, `GeoClass.txt`, forcing `Pobs.txt`/`Tobs.txt`/…, and the
  executable). The wrapper copies it `MultipleRuns` times into a temp directory
  and runs those copies in parallel.
- **Two objectives.** Each member is scored on **non-exceedance / reliability**
  and on a **streamflow error metric** (MAE/MSE/NSE). Selection keeps a
  non-dominated front (NSGA-II sorting + crowding), so calibration returns a
  *population*, not a single parameter set.
- **Probability bands.** The population's simulations are aggregated into
  predictive quantile bands (default
  `[0.001, 0.01, 0.025, 0.05, 0.15, …, 0.95, 0.975, 0.99, 0.999]`), giving the
  ensemble/probabilistic forecast.

---

## Quick start

```python
from pathlib import Path
import pandas as pd
from gpu_pso import GPU_PSO
from gpu import GPU
from error.errorOpenCL import Error
from conceptual.HYPE import HYPE

# Observed discharge for the Türkheim outlet (subbasin 50675)
y = pd.read_csv("examples/Qobs.txt", sep="\t", header=0,
                index_col="Date", parse_dates=True)
init, final = pd.to_datetime("1980-01-01"), pd.to_datetime("1989-12-31")
y = y.loc[(y.index >= init) & (y.index <= final)]

# HYPE model + GPU error model
Model = HYPE(MultipleRuns=8, records=y.index,
             calibration_parameters=["wcfc", "rrcs1", "ttmp", "cmlt", "preccorr"],
             random=True, normalization=True, log=True,
             HYPEfolder="examples/set7_germany_tuerkheim",
             outfile="results/0050675.txt")
Model.set_simulation_Dates(init, final)

opt = GPU_PSO(modelObject=Model, errorObject=Error(errorFunction="MAE"),
              variables=len(Model.parList[0]), population=30,
              inertia=0.2, c1=0.3, c2=0.2, c3=0.001, pBins=10, partial=0.1,
              forcePositive=False, transformWeights=False,
              forceNonExceedance=0.01, displayEach=2)

result, performance = opt.fit(y, epochs=5)   # short demo run
opt.modelObject.remove_tmpFiles()
opt.save("examples/my_calibration.pkl")
```

To forecast with the bundled pre-trained model instead, see
[`02_training_and_predicting`](notebooks/02_training_and_predicting.ipynb).

---

## Project structure

```
GPU_HYPE/
├── gpu.py                      # abstract GPU engine: fit/predict/save/load, Display/PlotGPU
├── gpu_pso.py                  # GPU_PSO — Particle-Swarm generate/select strategy
├── domination.py, crowding.py  # NSGA-II non-dominated sorting + crowding distance
├── conceptual/
│   └── HYPE.py                 # HYPE wrapper (runs the executable in parallel) + metaHYPE
├── error/
│   ├── errorOpenCL.py          # Error — OpenCL MAE/MSE objective
│   ├── errorNumpy.py           # NumPy fallback
│   └── evalMAE.cl, evalMSE.cl  # OpenCL kernels
├── new_error_model.py          # NewErrorModel — NumPy MAE/MSE/NSE objective
├── regression/                 # ANN regressor + activation kernels (alternative model)
├── functions.py                # band/quantile helpers used by the engine
├── functions2.py               # optional diagnostics (needs forecast_performance)
├── data.py                     # Data container for the ANN workflow
├── Trainer.py                  # end-to-end calibration script
├── DisplayPKL.py               # load and show a saved figure pickle
├── examples/                   # bundled Türkheim catchment + pre-trained model
│   ├── set7_germany_tuerkheim/ # HYPE setup folder (incl. the executable)
│   ├── Qobs.txt                # observed discharge (subbasin 50675)
│   ├── GPU_HYPE_2026-03-24_13h15.pkl          # pre-trained GPU_PSO model
│   └── BestAutomaticCalibration_0050675.txt   # HYPE's own best calibration (baseline)
├── Pickles/                    # small input/output sample for the ANN path
├── notebooks/                  # 00_setup_and_data … 03_results
├── tests/                      # smoke tests (skip without OpenCL / the HYPE exe)
├── AGENTS.md                   # conventions for AI coding agents (canonical)
├── CLAUDE.md                   # → points to AGENTS.md
├── .github/copilot-instructions.md # → points to AGENTS.md
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Running the tests

```bat
pytest tests/ -v
```

The tests import the package and load the bundled model; they **skip** when
OpenCL or the HYPE executable is unavailable.

---

## API reference

### `GPU_PSO` / `GPU`

| Member | Description |
|---|---|
| `GPU_PSO(modelObject, errorObject, variables, population=1000, epochs=400, bands=[...], inertia, c1, c2, c3, pBins, partial, ...)` | Build the Particle-Swarm optimiser around a model and an error model. |
| `fit(y, X=None, epochs=None, save=None)` | Calibrate. Returns `(result, performance)`; `result` has `parameters`, `fitness`, `states`, `aggregated`. |
| `predict(X=None, bands=None, postProcess=False, all=False)` | Forecast. For HYPE, re-runs the executable with the calibrated population and returns the probability bands. |
| `save(path)` / `GPU.load(path)` | Pickle / restore the whole optimiser. |
| `get_best_model()` | Deterministic prediction of the single best (lowest-error) member. |
| `get_par_distribution(save_path=None, show_plots=False)` | Distribution of the calibrated parameters across the population. |

### `HYPE` (`conceptual/HYPE.py`)

| Member | Description |
|---|---|
| `HYPE(MultipleRuns, records, calibration_parameters, random, normalization, log, HYPEfolder, outfile)` | Wrap a HYPE folder; sets up `MultipleRuns` parallel run folders. |
| `set_simulation_Dates(init_date, final_date, b_init=None)` | Rewrite `bdate`/`cdate`/`edate` in the run folders. |
| `init_tmpFiles()` / `remove_tmpFiles()` | Create / delete the temporary parallel run folders. |
| `compute()` | Run HYPE forward for the current parameter population. |

### Error models

| Class | Description |
|---|---|
| `Error(errorFunction="MAE")` *(`error/errorOpenCL.py`)* | OpenCL MAE/MSE objective; returns `(error, non_exceedance)`. |
| `NewErrorModel(errorFunction="MAE", non_exceedance_threshold=0.0, logQopt=False)` *(`new_error_model.py`)* | NumPy MAE/MSE/NSE objective with optional log-space and thresholding. |

### Visualisation (`gpu.py`)

| Member | Description |
|---|---|
| `Display(y, opt, bandBounds)` | Drives the training-time plots; `prepare(...)` logs metrics, `showPlots(...)` renders and saves `.pplot`/`.png`. |
| `PlotGPU(y, bands)` | The three panels: `pareto(fit, rejected)`, `qq(pValues)`, `timeseries(aggregated)`. |
| `Display.show_pickle(path)` | Reopen a saved `.pplot` figure (see [DisplayPKL.py](DisplayPKL.py)). |

---

## License

See [LICENSE](LICENSE) (GNU GPL v3). The bundled `HYPEwithoutPopup4All.exe` is the
HYPE model by [SMHI](https://hypeweb.smhi.se/) and remains subject to its own
licence terms.
