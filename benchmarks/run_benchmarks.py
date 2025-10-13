import argparse
import csv
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Tuple

import numpy as np

# Import the current implementation directly from the repo
from src.gsindex import geometrical_separability_index


@dataclass
class BenchResult:
    n_samples: int
    n_features: int
    n_classes: int
    repeats: int
    time_mean_s: float
    time_std_s: float
    time_min_s: float
    time_max_s: float
    peak_mem_kib: int
    gsi_example: float


def make_dataset(n_samples: int, n_features: int, n_classes: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Create class centers spaced apart to avoid degenerate overlap; not required for benchmarking
    centers = rng.uniform(-5, 5, size=(n_classes, n_features)) + np.arange(n_classes)[:, None] * 5.0
    labels = rng.integers(0, n_classes, size=n_samples)
    # Add moderate cluster spread
    X = centers[labels] + rng.normal(scale=1.0, size=(n_samples, n_features))
    return X, labels.astype(object)  # object labels stress label handling (strings/numbers compatible)


def time_once(X: np.ndarray, y: np.ndarray) -> Tuple[float, int, float]:
    # Track Python allocations; peak_mem_kib is indicative, not total native memory
    tracemalloc.start()
    t0 = time.perf_counter()
    gsi_val = geometrical_separability_index(X, y)
    dt = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kib = int(peak / 1024)
    return dt, peak_kib, gsi_val


def run_bench(n_samples: int, n_features: int, n_classes: int, repeats: int, base_seed: int = 42) -> BenchResult:
    # Use a fixed dataset across repeats to reduce noise
    X, y = make_dataset(n_samples, n_features, n_classes, seed=base_seed)
    times: List[float] = []
    peaks: List[int] = []
    gsi_example: float = 0.0
    for r in range(repeats):
        dt, peak_kib, gsi_val = time_once(X, y)
        times.append(dt)
        peaks.append(peak_kib)
        if r == 0:
            gsi_example = float(gsi_val)

    return BenchResult(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        repeats=repeats,
        time_mean_s=statistics.fmean(times),
        time_std_s=(statistics.pstdev(times) if repeats > 1 else 0.0),
        time_min_s=min(times),
        time_max_s=max(times),
        peak_mem_kib=max(peaks),  # report max peak observed
        gsi_example=gsi_example,
    )


def parse_int_list(values: Iterable[str]) -> List[int]:
    return [int(v) for v in values]


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark geometrical_separability_index")
    p.add_argument("--sizes", nargs="*", default=[], help="List of sample sizes, e.g., 200 500 1000")
    p.add_argument("--features", nargs="*", default=[], help="List of feature counts, e.g., 10 50 100")
    p.add_argument("--classes", type=int, default=2, help="Number of classes (>=2)")
    p.add_argument("--repeats", type=int, default=3, help="Repeats per configuration")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    p.add_argument("--quick", action="store_true", help="Run a quick preset (sizes=100 500 1000, features=10)")

    args = p.parse_args(argv)

    if args.quick:
        sizes = [100, 500, 1000]
        feats = [10]
    else:
        sizes = parse_int_list(args.sizes) if args.sizes else [200, 500, 1000]
        feats = parse_int_list(args.features) if args.features else [10]

    if args.classes < 2:
        raise SystemExit("--classes must be >= 2")

    grid = list(product(sizes, feats))
    results: List[BenchResult] = []
    print("Benchmarking geometrical_separability_index")
    print(f"Classes: {args.classes} | Repeats: {args.repeats}")
    print("n_samples\tn_features\tt_mean[s]\tt_std[s]\tt_min[s]\tt_max[s]\tpeak_mem[KiB]\tgsi")
    for ns, nf in grid:
        try:
            res = run_bench(ns, nf, args.classes, args.repeats)
        except MemoryError:
            print(f"{ns}\t{nf}\tOOM\tOOM\tOOM\tOOM\tOOM\t-")
            continue
        results.append(res)
        print(
            f"{res.n_samples}\t{res.n_features}\t"
            f"{res.time_mean_s:.6f}\t{res.time_std_s:.6f}\t{res.time_min_s:.6f}\t{res.time_max_s:.6f}\t"
            f"{res.peak_mem_kib}\t{res.gsi_example:.6f}"
        )

    if args.csv and results:
        fieldnames = [
            "n_samples",
            "n_features",
            "n_classes",
            "repeats",
            "time_mean_s",
            "time_std_s",
            "time_min_s",
            "time_max_s",
            "peak_mem_kib",
            "gsi_example",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.__dict__)
        print(f"\nCSV written to: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

