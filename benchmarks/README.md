Benchmarks for gsindex

Overview
- This folder provides a lightweight, standalone benchmark script for the current implementation of `geometrical_separability_index`.
- It measures wall-clock time (seconds) and Python-level peak memory (via `tracemalloc`) across dataset sizes and feature counts.

How to Run
- Quick run (small sizes):
  - `python benchmarks/run_benchmarks.py --quick`
- Custom run (examples):
  - `python benchmarks/run_benchmarks.py --sizes 200 500 1000 --features 10 50 --classes 2 --repeats 3`
  - `python benchmarks/run_benchmarks.py --sizes 1000 2000 --features 20 --classes 3 --repeats 5 --csv benchmarks/results.csv`

Notes
- The script imports the function from `src.gsindex` to benchmark the current code in-place.
- Memory tracking uses `tracemalloc` which measures Python allocations; native allocations from NumPy/SciPy may not be fully captured.
- Ensure SciPy and NumPy are installed (`pip install -e .[test]` suffices in this repo).
- For very large sizes (O(n^2) memory for distances), the run can be slow or exhaust memory.

