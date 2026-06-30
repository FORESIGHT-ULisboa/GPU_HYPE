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
| **Models** | `HYPE` (runs the HYPE executable in parallel) |
| **Error models** | `NewErrorModel` (NumPy MAE/MSE/NSE) |
| **Probabilistic output** | population → predictive uncertainty bands (default 16 levels) |
| **Multi-objective** | reliability (non-exceedance) vs error, with NSGA-II sorting + crowding |
| **Visualisation** | `Display` / `PlotGPU` — Pareto front, PIT/Q-Q reliability, hydrograph bands |
| **Persistence** | `save` / `load` the whole optimiser as a pickle |

---

## Installation

### 1 · Create a conda environment

```bat
conda env create -f environment.yml
conda activate gpu_hype
```

### 2 · Install the package and dependencies

From the repository root:

```bat
pip install -e ".[dev]"
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

The runnable end-to-end pipeline targets **Windows**:

- **Windows.** The HYPE model is invoked as `HYPEwithoutPopup4All.exe`
  (bundled in each HYPE folder) through `subprocess`.
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
  `[0.01, 0.025, 0.05, 0.15, …, 0.95, 0.975, 0.99]`), giving the probabilistic forecast.

---

## Quick start

```python
from pathlib import Path
import pandas as pd
from gpu_model.gpu_pso import GPU_PSO
from gpu_model.gpu import GPU
from error.error_model import NewErrorModel
from conceptual.HYPE import HYPE

# Observed discharge for the Türkheim outlet (subbasin 50675)
y = pd.read_csv("demo_model/Qobs.txt", sep="\t", header=0,
                index_col="Date", parse_dates=True)
init, final = pd.to_datetime("2001-01-01"), pd.to_datetime("2001-06-31")
y = y.loc[(y.index >= init) & (y.index <= final)]

# HYPE model + GPU error model
Model = HYPE(MultipleRuns=8, records=y.index,
             calibration_parameters=["wcfc", "rrcs1", "ttmp", "cmlt", "preccorr"],
             random=True, normalization=True, log=True,
             HYPEfolder="demo_model/HYPE_setup_folder",
             outfile="results/0003587.txt")
             parameter_bounds = { #Parameter bound
                'wcfc': [0.0001, 1, True, 'substitute'], 
                'rrcs1': [0.0001, 1, True, 'substitute'],
                'rrcs2': [0.0001, 1, True, 'substitute'],
                'preccorr': [0.0001, 1, True, 'substitute'],
                'ttmp': [-3, 5, False, 'substitute'],
            },
          
Model.set_simulation_Dates(init, final)

opt = GPU_PSO(modelObject=Model, errorObject=NewErrorModel(errorFunction="MAE"),
              variables=len(Model.parList[0]), population=30,
              inertia=0.2, c1=0.3, c2=0.2, c3=0.001, pBins=10, partial=0.1,
              forcePositive=False, transformWeights=False,
              forceNonExceedance=0.01, displayEach=2)

result, performance = opt.fit(y, epochs=5)   # short demo run
opt.modelObject.remove_tmpFiles()
opt.save("notebook_results/my_calibration.pkl")
```

---

## Project structure

```
GPU_HYPE/
├── gpu_model/
    ├── gpu_pso.py                  # GPU_PSO — Particle-Swarm generate/select strategy
    ├── gpu.py                      # Display / PlotGPU / GPU core
    ├── domination.py, crowding.py  # NSGA-II non-dominated sorting + crowding distance
    ├── functions.py                # band/quantile helpers used by the engine
    ├── functions2.py               # optional diagnostics (needs forecast_performance)            
├── conceptual/
│   └── HYPE.py                     # HYPE wrapper (runs the executable in parallel) + metaHYPE
├── error/
│   └── error_model.py          # NewErrorModel — NumPy MAE/MSE/NSE objective

├── data.py                     # Data container for the ANN workflow
├── Trainer.py                  # end-to-end calibration script
├── DisplayPKL.py               # load and show a saved figure pickle
├── examples/                   # bundled Türkheim catchment + pre-trained model
│   ├── set7_germany_tuerkheim/ # HYPE setup folder (incl. the executable)
│   ├── Qobs.txt                # observed discharge (subbasin 50675)
│   ├── GPU_HYPE_2026-03-24_13h15.pkl          # pre-trained GPU_PSO model
│   └── BestAutomaticCalibration_0050675.txt   # HYPE's own best calibration (baseline)
├── notebooks/                  # 00_setup_and_data 01_calibration_and_prediction 
├── notebooks_results/          # Store the results of the notebooks
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

### Error model

| Class | Description |
|---|---|
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
HYPE model by [SMHI](https://hypeweb.smhi.se/) with slight modifications and remains subject to its own
licence terms.
