# AGENTS.md — conventions for AI coding agents

This file is the canonical instruction set for AI agents (Claude Code, GitHub
Copilot, etc.) working in this repository. `CLAUDE.md` and
`.github/copilot-instructions.md` are thin pointers to this file.

## What this project is

`GPU_HYPE` calibrates the **HYPE** hydrological model with a GPU-assisted,
multi-objective evolutionary optimiser and turns the resulting parameter
population into a **probabilistic (ensemble) streamflow forecast**. It is the
research code published alongside the paper; the goal of this repo is to be a
runnable, well-documented companion, **not** a refactor target.

The public surface is the `GPU_PSO` optimiser in
[gpu_pso.py](gpu_pso.py) (a Particle-Swarm subclass of the abstract `GPU` engine
in [gpu.py](gpu.py)), driven by two pluggable objects:

- a **model** — `HYPE` ([conceptual/HYPE.py](conceptual/HYPE.py)), which runs the
  HYPE executable, or `ANN` ([regression/annOpenCL.py](regression/annOpenCL.py)),
  an OpenCL neural-network regressor; and
- an **error model** — `Error`
  ([error/errorOpenCL.py](error/errorOpenCL.py), OpenCL MAE/MSE kernels) or
  `NewErrorModel` ([new_error_model.py](new_error_model.py), NumPy MAE/MSE/NSE).

The end-to-end workflow — set up `HYPE` + an error model, wrap them in `GPU_PSO`,
`fit`, `save`, `load`, `predict`, then display — is shown in
[Trainer.py](Trainer.py) and the [notebooks](notebooks/).

## Environment

- The project runs in a **conda environment with Python 3.8** (the same
  interpreter used to produce the bundled model). Use it for the package, the
  tests, and the notebooks (register it as a Jupyter kernel).
- Hint (a developer machine; **path will differ on other computers** — do not
  hard-code it): the interpreter has been seen at
  `C:\Users\<your-user>\.conda\envs\Python38\python.exe`. `conda` may not be on
  `PATH`; if so, call that `python.exe` directly or activate the env in a
  conda-aware shell.
- Setup: `pip install -e ".[dev]"`. The advanced verification metrics and
  hydrograph diagnostics in [functions2.py](functions2.py) reuse the companion
  `forecast_performance` package — install it with `pip install -e ".[metrics]"`
  (or from the sibling repo). The core calibration/training/prediction path does
  **not** import it.
- **Runtime requirements (important):**
  - **OpenCL.** Importing `gpu_pso` constructs an OpenCL context (the
    `Error()`/`ANN()` defaults in `GPU_PSO.__init__` are evaluated at import).
    A working OpenCL platform — GPU or a CPU ICD — must be present.
  - **Windows + the HYPE executable.** `HYPE` shells out to
    `HYPEwithoutPopup4All.exe` (bundled in each HYPE folder) via `subprocess`,
    so the runnable HYPE path is Windows-only.
- Run tests with `pytest tests/ -v` (they skip themselves when OpenCL or the
  HYPE executable is unavailable).

## Architecture (flat layout — keep it)

This is research code with a deliberately flat module layout. **Do not** convert
it into a nested package, rewrite `from gpu import *` imports, or rename the
public classes; downstream pickles and the paper reference them by name.

- [gpu.py](gpu.py) — the abstract `GPU` engine: population init, the
  evaluation/iteration loop, multi-objective bookkeeping, band aggregation,
  `fit` / `predict` / `save` / `load`, and the `Display` / `PlotGPU`
  visualisation classes.
- [gpu_pso.py](gpu_pso.py) — `GPU_PSO`, the Particle-Swarm `_generate` / `_select`
  strategy (other strategies are possible by overriding those two hooks).
- [domination.py](domination.py) / [crowding.py](crowding.py) — NSGA-II style
  non-dominated sorting and crowding distance used during selection.
- [conceptual/HYPE.py](conceptual/HYPE.py) — the `HYPE` wrapper: writes `par.txt`,
  spawns `MultipleRuns` parallel copies of the HYPE folder under a temp dir, runs
  the executable, and reads the per-basin output (`outfile`). `metaHYPE` adds a
  linear post-processing correction. The model is run forward by `compute()`.
- [error/](error/) — `Error` plus the `evalMAE.cl` / `evalMSE.cl` OpenCL kernels
  (loaded via `pkg_resources`); [error/errorNumpy.py](error/errorNumpy.py) is the
  CPU fallback.
- [regression/](regression/) — the `ANN` regressor and its activation kernels
  (`ann*.cl`), the alternative non-HYPE model.
- [functions.py](functions.py) — band/quantile helpers (`processBands`,
  `mpld3Correct`, …) used by the engine.
- [functions2.py](functions2.py) — **optional** higher-level diagnostics
  (`plot_prediction`, `calculate_metrics`, `aggregated_to_df`,
  `median_prediction`). This is the only module that imports `performance`.
- [data.py](data.py) — the `Data` container for the ANN/regression workflow
  (seasonal splits, train/validation periods).

## The calibration / training / prediction API

- **Construct** the model and error objects, then the optimiser:
  ```python
  Model = HYPE(MultipleRuns=12, records=index, calibration_parameters=[...],
               random=True, normalization=True, log=True,
               HYPEfolder="examples/set7_germany_tuerkheim",
               outfile="results/0050675.txt")
  Model.set_simulation_Dates(init_date, final_date)
  error_model = Error(errorFunction="MAE")          # or NewErrorModel(...)
  opt = GPU_PSO(modelObject=Model, errorObject=error_model,
                variables=len(Model.parList[0]), population=250, ...)
  ```
- **Train / calibrate:** `result, performance = opt.fit(y, epochs=..., save=path)`.
  `fit` returns a tuple: `result` is a dict with keys `parameters` (the calibrated
  population), `fitness` (`[non_exceedance, log_error]` per member), `states`, and
  `aggregated`; `performance` is the per-epoch log produced by `Display`. The two
  objectives are **reliability / non-exceedance** vs the **error metric**.
- **Persist:** `opt.save(path)` pickles the whole optimiser;
  `GPU_PSO.load(path)` restores it (and re-aliases the legacy `HYPE` module).
- **Predict (HYPE):** a loaded model has no temp folders, so re-create them
  before predicting and point the model at the bundled folder:
  ```python
  m = GPU.load(path)
  m.modelObject.DefaultFolderLoc = Path("examples/set7_germany_tuerkheim")
  m.modelObject.init_tmpFiles()
  m.modelObject.set_simulation_Dates(init, final, b_init=init)
  m.modelObject.records = pd.date_range(init, final, freq="D")
  aggregated = m.predict()                 # (n_steps, n_bands) probability bands
  m.modelObject.remove_tmpFiles()
  ```
  `predict()` re-runs HYPE with the calibrated population over whatever dates the
  temp folders are set to. The `predict(X, ...)` signature carries `X` for the
  ANN path; the HYPE branch ignores it.

## Data formats and the HYPE folder

- A **HYPE folder** (e.g. [examples/set7_germany_tuerkheim](examples/set7_germany_tuerkheim))
  is a standard HYPE setup: `info.txt` (simulation dates + `basinoutput` settings),
  `par.txt` (parameters), `GeoData.txt` / `GeoClass.txt` (subbasin topology and
  land-use classes), forcing (`Pobs.txt`, `Tobs.txt`, `TMAXobs.txt`, `TMINobs.txt`),
  and the `HYPEwithoutPopup4All.exe` executable. HYPE writes its basin output to
  `results/00<SUBID>.txt`.
- **Observations** (`Qobs`) are a date-indexed table; the bundled
  [examples/Qobs.txt](examples/Qobs.txt) is tab-separated with a `Date` column and
  a `50675` discharge column (the Türkheim outlet). Read with
  `pd.read_csv(..., sep="\t", index_col="Date", parse_dates=True)`.
- `set_simulation_Dates(init, final, b_init=None)` rewrites `bdate`/`cdate`/`edate`
  in the temp folders' `info.txt`; the model's `records` index must match the
  dates being simulated.

## The bundled example

- [examples/set7_germany_tuerkheim](examples/set7_germany_tuerkheim) — a HYPE
  setup for the Türkheim sub-catchment (outlet subbasin **50675**), forcing
  1980–2019.
- [examples/Qobs.txt](examples/Qobs.txt) — observed discharge for subbasin 50675.
- [examples/GPU_HYPE_2026-03-24_13h15.pkl](examples/GPU_HYPE_2026-03-24_13h15.pkl)
  — a pre-trained `GPU_PSO` (HYPE model, 250-member population, 17 calibrated
  parameters, `NewErrorModel`), so the prediction/results notebooks run without a
  long calibration. Calibrated on **1980-01-01 → 1999-12-31**; validation period
  starts **2000-01-01**.
- [examples/BestAutomaticCalibration_0050675.txt](examples/BestAutomaticCalibration_0050675.txt)
  — HYPE's own best single calibration, used as a deterministic baseline in the
  results notebook.
- [Pickles/](Pickles/) — a small input/output sample for the ANN/regression path.

## Notebooks

Numbered, self-contained walkthroughs in [notebooks/](notebooks/) that run with
the working directory at the repo root:

- `00_setup_and_data` — environment & OpenCL check, the HYPE folder and data
  formats.
- `01_calibration` — the calibration tool: build `HYPE` + `Error` + `GPU_PSO`
  and run a short `fit`.
- `02_training_and_predicting` — load the pre-trained model and `predict` a
  probabilistic ensemble.
- `03_results` — display hydrographs with uncertainty bands, the PIT/Q-Q
  reliability diagram and Pareto front (`Display` / `PlotGPU`), and skill
  metrics (with an optional `performance`-based comparison vs HYPE's own
  calibration).

## Style

- Keep the flat layout and the existing public names. Prefer additive,
  repo-relative changes over edits to the research modules; if a research module
  genuinely must change, flag it rather than silently rewriting it.
- Notebooks resolve the repo root and use paths under `examples/`; don't
  hard-code machine-specific absolute paths.
