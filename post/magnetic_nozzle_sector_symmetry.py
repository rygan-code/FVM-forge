"""Measure six-sector rotational covariance in magnetic-nozzle PLT output."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np


BLOCKS = 6
SCALARS = ("rho", "p", "T", "phi", "psi")
VECTORS = (("velocity", "u", "v", "w"), ("magnetic", "Bx", "By", "Bz"))


def steps_in(plt_dir: Path, blocks: int) -> list[int]:
    pattern = re.compile(r"plt-(\d+)-b0\.h5$")
    candidates = sorted(
        int(match.group(1))
        for path in plt_dir.glob("plt-*-b0.h5")
        if (match := pattern.match(path.name))
    )
    return [
        step for step in candidates
        if all((plt_dir / f"plt-{step}-b{block}.h5").is_file()
               for block in range(blocks))
    ]


def load_step(plt_dir: Path, step: int, blocks: int) -> list[dict[str, np.ndarray]]:
    fields = set(SCALARS)
    for _, x_name, y_name, z_name in VECTORS:
        fields.update((x_name, y_name, z_name))
    result: list[dict[str, np.ndarray]] = []
    for block in range(blocks):
        with h5py.File(plt_dir / f"plt-{step}-b{block}.h5", "r") as handle:
            result.append({name: np.asarray(handle[name], dtype=np.float64)
                           for name in fields})
    return result


def relative_linf(reference: np.ndarray, value: np.ndarray) -> tuple[float, tuple[int, ...]]:
    difference = np.abs(value - reference)
    location = tuple(int(index) for index in np.unravel_index(np.argmax(difference), difference.shape))
    scale = max(float(np.max(np.abs(reference))), np.finfo(np.float64).tiny)
    return float(np.max(difference) / scale), location


def compare(data: list[dict[str, np.ndarray]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    angles = np.arange(1, len(data), dtype=np.float64) * (2.0 * np.pi / 6.0)
    for name in SCALARS:
        errors = []
        locations = []
        for sector, angle in enumerate(angles, start=1):
            error, location = relative_linf(data[0][name], data[sector][name])
            errors.append(error)
            locations.append(location)
        summary[name] = (max(errors), errors, locations)

    for label, x_name, y_name, z_name in VECTORS:
        errors = []
        locations = []
        for sector, angle in enumerate(angles, start=1):
            cosine, sine = np.cos(angle), np.sin(angle)
            x = data[sector][x_name]
            y = cosine * data[sector][y_name] + sine * data[sector][z_name]
            z = -sine * data[sector][y_name] + cosine * data[sector][z_name]
            reference = np.stack((data[0][x_name], data[0][y_name], data[0][z_name]))
            value = np.stack((x, y, z))
            error, location = relative_linf(reference, value)
            errors.append(error)
            locations.append(location)
        summary[label] = (max(errors), errors, locations)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--blocks", type=int, default=BLOCKS)
    args = parser.parse_args()
    plt_dir = args.run_dir / "PLT"
    steps = steps_in(plt_dir, args.blocks)
    if not steps:
        raise RuntimeError(f"no complete PLT steps found in {plt_dir}")

    print("step,field,max_relative_linf,sector_errors,worst_locations")
    for step in steps:
        summary = compare(load_step(plt_dir, step, args.blocks))
        for field, (maximum, errors, locations) in summary.items():
            print(f"{step},{field},{maximum:.9e},"
                  f"{','.join(f'{value:.3e}' for value in errors)},"
                  f"{locations}")


if __name__ == "__main__":
    main()
