Project Guide for Agents

Overview
- Purpose: Compute the Geometrical Separability Index (GSI) for labeled datasets.
- Status: Minimal implementation present; expanded tests and benchmarks added; packaging modernized to Python 3.12+.
- Key tradeoff: Current algorithm builds a full pairwise distance matrix (O(n^2) time and memory).

Repo Layout
- `src/gsindex.py`: Implementation of `geometrical_separability_index` (no input validation; Euclidean via SciPy `cdist`).
- `src/__init__.py`: Exposes `__version__` using `importlib.metadata` when installed.
- `tests/test_gsindex.py`: Unit tests (unittest-based) including edge cases and multiclass.
- `benchmarks/`: Lightweight benchmark script + README.
- `Taskfile.yaml`: Task runner shortcuts (venv + benchmark commands).
- `.github/workflows/`: CI workflows (integration tests, build, publish).
- `pyproject.toml`, `setup.cfg`, `setup.py`: Packaging configuration (PEP 517 + setuptools_scm).
- `TODO.md`: Roadmap with packaging, API, testing, perf, and CI improvements.

Implementation Notes (Current Behavior)
- Function: `geometrical_separability_index(matrix, labels)`
  - Computes full pairwise distances with `scipy.spatial.distance.cdist`.
  - Uses `argsort(axis=0)`, then compares each sample’s label with its nearest neighbor’s label (skipping self).
  - Returns ratio in [0,1].
- Inputs/validation:
  - Expects 2D array-like `matrix` and 1D `labels` with length `n_samples`.
  - No explicit validation; errors bubble from SciPy/NumPy (e.g., 1D matrix -> ValueError; single sample -> IndexError).
- Ties/duplicates:
  - Behavior is whatever NumPy/SciPy ordering yields; no explicit tie policy implemented or documented.

Environment & Tooling
- Supported Python: >= 3.12 (see `setup.cfg: python_requires`).
- Dependencies (runtime): `numpy>=1.26`, `scipy>=1.12`.
- Dev/test extras: `pytest>=8`, `flake8>=7`.
- Task runner: `go-task` (`Taskfile.yaml`).

Setup & Installation
- Local venv and editable install via Taskfile:
  - `task venv:create`
  - `task venv:install`
- Manual (without Taskfile):
  - `python -m venv .venv`
  - Unix: `. .venv/bin/activate`  |  Windows: `.venv\Scripts\activate`
  - `pip install --upgrade pip`
  - `pip install -e .[test]`

Running Tests
- With unittest:
  - `python -m unittest -v`
- With pytest (installed via extras):
  - `pytest -q`
- What’s covered:
  - Perfect/mixed/no separation scenarios (binary labels).
  - Single-sample IndexError (current behavior).
  - Duplicates within class (GSI=1.0) and identical cross-class pairs (GSI=0.0).
  - Multiclass, numeric labels, optional pandas inputs (skips if pandas missing).
  - Non-finite values: ensures returns a ratio in [0,1] without crashing.
  - Shape errors: mismatched label length -> IndexError; non-2D matrix -> ValueError.

Benchmarks
- Script: `benchmarks/run_benchmarks.py`
  - Measures wall-clock time per run and Python-level peak memory (`tracemalloc`).
  - Deterministic synthetic datasets by size/features/classes; repeated runs per config.
  - Outputs a table to stdout and optionally CSV.
- Quick run:
  - `task bench:quick`
  - or: `python benchmarks/run_benchmarks.py --quick`
- Custom run examples:
  - `task bench:run SIZES='200 500 1000' FEATURES='10 50' CLASSES=2 REPEATS=3`
  - `python benchmarks/run_benchmarks.py --sizes 1000 2000 --features 20 --classes 3 --repeats 5 --csv benchmarks/results.csv`
- Caveat: For large `n_samples`, the O(n^2) distance matrix can be slow or OOM.

CI/CD
- Integrate workflow: `.github/workflows/integrate.yaml`
  - Matrix: OS = ubuntu/windows/macos; Python = 3.12, 3.13.
  - Steps: checkout, setup-python, `pip install -e .[test]`, flake8 (basic + relaxed), pytest.
- Build/publish workflows: `build.yaml`, `publish.yaml` (use `3.x`).
  - Can be pinned to a specific 3.12+ version if needed.

Known Limitations
- Performance: Full pairwise distances cause O(n^2) time/memory.
- Validation: No explicit input checking; unclear error messages may surface from SciPy/NumPy.
- Ties: No deterministic tie-breaking policy is documented or configurable.
- Packaging layout: Module is `src/gsindex.py` rather than `src/gsindex/__init__.py` package structure (see TODO).

Roadmap (from TODO.md – highlights)
- Packaging:
  - Adopt proper `src/` layout with `src/gsindex/` package and update imports/tests/README accordingly.
  - Move metadata to PEP 621 in `pyproject.toml`; minimize `setup.cfg`, drop `setup.py`.
  - Configure `setuptools_scm` under `[tool.setuptools_scm]`.
- API & Validation:
  - Add type hints and comprehensive docstring.
  - Validate shapes and values; raise `ValueError` with clear messages.
  - Optional distance metrics (`metric`, `metric_kwargs`), optional per-class GSI.
  - Decide and document tie-breaking policy; optionally make it configurable.
- Performance:
  - Avoid full sort/matrix: `argpartition`, `cKDTree`, or `sklearn.NearestNeighbors` for large datasets.
  - Benchmark and document guidance by dataset size.
- Testing:
  - Expand coverage as features are added (metrics, validation messages, pandas support).
  - Optional property-based tests with `hypothesis`.
- Docs & CI:
  - README: formal definition, complexity, examples; badges; CHANGELOG; CONTRIBUTING; CITATION.
  - CI: cache deps; type checks (mypy); formatter/linter (ruff/black) in CI; coverage reporting.

Common Commands
- Lint (CI-like quick run):
  - `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics`
- Tests:
  - `pytest -q` or `python -m unittest -v`
- Benchmarks:
  - `task bench:quick`
  - `task bench:run SIZES='200 500 1000' FEATURES='10' CLASSES=2 REPEATS=3 CSV=benchmarks/results.csv`
- CI locally (requires `act`):
  - `task workflow:integrate`

Assumptions for Future Work
- Keep Python >= 3.12 (update classifiers/matrix if adjusting).
- Preserve current behavior in tests unless explicitly changing (e.g., single-sample should raise ValueError later—update tests accordingly).
- When migrating packaging to `gsindex/` package, update imports in tests and README to `from gsindex import geometrical_separability_index`.

