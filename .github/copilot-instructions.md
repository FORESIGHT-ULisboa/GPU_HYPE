See [AGENTS.md](../AGENTS.md) for this project's conventions, architecture, the
calibration/training/prediction API, the runtime environment (OpenCL + the HYPE
executable), and how the bundled Türkheim example is wired. It is the single
source of truth for AI coding agents working in this repository.

Key reminders:
- This is research code published alongside a paper. **Do not refactor the flat
  module layout** (`gpu.py`, `gpu_pso.py`, `conceptual/HYPE.py`, `error/`,
  `regression/`) or rename the public API (`GPU_PSO`, `HYPE`, `Error`,
  `NewErrorModel`, `Display`).
- It runs on **Windows with an OpenCL platform**: importing `gpu_pso` builds an
  OpenCL context, and `conceptual/HYPE.py` shells out to
  `HYPEwithoutPopup4All.exe` inside each run folder.
- Notebooks run with the working directory at the repo root and use paths under
  `examples/`. The companion `performance` package is only needed for the
  optional `functions2` metrics/plots.
