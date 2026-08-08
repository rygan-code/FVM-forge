"""Convert the SI magnetic-nozzle snapshots to the paper's dimensionless form.

The Lorzel--Mikellides paper reports dimensional performance values but uses
the standard stagnation/throat normalizations in its diagnostics.  This tool
keeps the solver output in SI and performs the conversion only in postprocess.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np


MU0 = 4.0 * math.pi * 1.0e-7
GAMMA = 5.0 / 3.0
RHO0 = 5.0e-5
T0 = 100.0 * 11604.51812
P0 = 0.355e6
B0 = 0.944
L_DOMAIN = 1.0
R_THROAT = 0.11
A_THROAT = math.pi * R_THROAT**2
V_PRESSURE = math.sqrt(P0 / RHO0)
MASS_SCALE = math.sqrt(RHO0 * P0) * A_THROAT
THRUST_SCALE = P0 * A_THROAT

PAPER_TARGETS = {
    "mass_flow_g_s": 55.2,
    "thrust_n": 8650.0,
    "core_mach": 2.80,
    "core_exhaust_velocity_m_s": 157960.0,
}


def cell_centers(values: np.ndarray) -> np.ndarray:
    return (
        values[:-1, :-1, :-1]
        + values[1:, :-1, :-1]
        + values[:-1, 1:, :-1]
        + values[:-1, :-1, 1:]
        + values[1:, 1:, :-1]
        + values[1:, :-1, 1:]
        + values[:-1, 1:, 1:]
        + values[1:, 1:, 1:]
    ) / 8.0


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        # Julia writes column-major arrays; h5py exposes (z, y, x).
        nodes = [np.asarray(handle[name]).transpose(2, 1, 0)
                 for name in ("x", "y", "z")]
    return tuple(cell_centers(values) for values in nodes)


def read_field(path: Path, name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[name]).transpose(2, 1, 0)


def face_areas(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    points = np.stack((x[-1], y[-1], z[-1]), axis=-1)
    p00 = points[:-1, :-1]
    p10 = points[1:, :-1]
    p11 = points[1:, 1:]
    p01 = points[:-1, 1:]
    vector_area = 0.5 * (
        np.cross(p10 - p00, p11 - p00)
        + np.cross(p11 - p00, p01 - p00)
    )
    return np.linalg.norm(vector_area, axis=-1)


def isentropic_ratios(mach: float) -> dict[str, float]:
    temperature = 1.0 + 0.5 * (GAMMA - 1.0) * mach * mach
    return {
        "rho0_over_rho": temperature ** (1.0 / (GAMMA - 1.0)),
        "T0_over_T": temperature,
        "p0_over_p": temperature ** (GAMMA / (GAMMA - 1.0)),
    }


def discover_steps(plt_dir: Path, blocks: int) -> list[int]:
    steps = []
    for path in plt_dir.glob("plt-*-b0.h5"):
        try:
            step = int(path.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if all((plt_dir / f"plt-{step}-b{bid}.h5").is_file()
               for bid in range(blocks)):
            steps.append(step)
    return sorted(set(steps))


def analyze_step(plt_dir: Path, mesh_dir: Path, step: int, blocks: int) -> dict[str, float]:
    mass_flow = 0.0
    thrust = 0.0
    core_speed_sum = 0.0
    core_mach_sum = 0.0
    core_weight = 0.0
    core_rho_min = math.inf
    core_rho_max = -math.inf
    core_b_min = math.inf
    core_b_max = -math.inf

    for block in range(blocks):
        mesh_path = mesh_dir / f"mesh_b{block}.h5"
        with h5py.File(mesh_path, "r") as handle:
            x_nodes = np.asarray(handle["x"]).transpose(2, 1, 0)
            y_nodes = np.asarray(handle["y"]).transpose(2, 1, 0)
            z_nodes = np.asarray(handle["z"]).transpose(2, 1, 0)
        x = cell_centers(x_nodes)
        y = cell_centers(y_nodes)
        z = cell_centers(z_nodes)
        area = face_areas(x_nodes, y_nodes, z_nodes)
        rho = read_field(plt_dir / f"plt-{step}-b{block}.h5", "rho")
        u = read_field(plt_dir / f"plt-{step}-b{block}.h5", "u")
        v = read_field(plt_dir / f"plt-{step}-b{block}.h5", "v")
        w = read_field(plt_dir / f"plt-{step}-b{block}.h5", "w")
        bx = read_field(plt_dir / f"plt-{step}-b{block}.h5", "Bx")
        by = read_field(plt_dir / f"plt-{step}-b{block}.h5", "By")
        bz = read_field(plt_dir / f"plt-{step}-b{block}.h5", "Bz")

        exit_area = area
        exit_rho = rho[-1]
        exit_u = u[-1]
        mass_flow += float(np.sum(exit_rho * exit_u * exit_area))
        thrust += float(np.sum(exit_rho * exit_u * exit_u * exit_area))

        # Block 0 is the Cartesian-like center block.  Use it alone for the
        # core-flow diagnostic; the surrounding blocks represent the current
        # layer and should not be counted as additional centerline samples.
        if block == 0:
            radius = np.sqrt(y[-1] ** 2 + z[-1] ** 2)
            core_index = np.unravel_index(np.argmin(radius), radius.shape)
            core_rho = float(exit_rho[core_index])
            core_velocity = float(np.sqrt(
                u[-1][core_index] ** 2 + v[-1][core_index] ** 2
                + w[-1][core_index] ** 2
            ))
            core_pressure = float(read_field(
                plt_dir / f"plt-{step}-b{block}.h5", "p"
            )[-1][core_index])
            core_sound = math.sqrt(GAMMA * core_pressure / core_rho)
            core_area = float(exit_area[core_index])
            core_speed_sum += core_velocity * core_area
            core_mach_sum += (core_velocity / core_sound) * core_area
            core_weight += core_area
        core_rho_min = min(core_rho_min, float(np.min(exit_rho)))
        core_rho_max = max(core_rho_max, float(np.max(exit_rho)))
        bmag = np.sqrt(bx[-1] ** 2 + by[-1] ** 2 + bz[-1] ** 2)
        core_b_min = min(core_b_min, float(np.min(bmag)))
        core_b_max = max(core_b_max, float(np.max(bmag)))

    return {
        "step": step,
        "mass_flow_kg_s": mass_flow,
        "mass_flow_hat": mass_flow / MASS_SCALE,
        "thrust_n": thrust,
        "thrust_hat": thrust / THRUST_SCALE,
        "core_mach": core_mach_sum / core_weight,
        "core_exhaust_velocity_m_s": core_speed_sum / core_weight,
        "core_exhaust_velocity_hat": (core_speed_sum / core_weight) / V_PRESSURE,
        "rho_hat_min": core_rho_min / RHO0,
        "rho_hat_max": core_rho_max / RHO0,
        "B_hat_min": core_b_min / B0,
        "B_hat_max": core_b_max / B0,
        "beta0": 2.0 * MU0 * P0 / (B0 * B0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("mesh_dir", type=Path)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    plt_dir = args.run_dir / "PLT"
    steps = discover_steps(plt_dir, args.blocks)
    if not steps:
        raise RuntimeError(f"no complete snapshots found in {plt_dir}")

    results = [analyze_step(plt_dir, args.mesh_dir, step, args.blocks)
               for step in steps]
    print(json.dumps({
        "reference_scales": {
            "rho0_kg_m3": RHO0,
            "p0_pa": P0,
            "T0_K": T0,
            "B0_t": B0,
            "mu0": MU0,
            "throat_radius_m": R_THROAT,
            "throat_area_m2": A_THROAT,
            "velocity_m_s": V_PRESSURE,
            "mass_flow_kg_s": MASS_SCALE,
            "thrust_n": THRUST_SCALE,
            "initial_beta": 2.0 * MU0 * P0 / (B0 * B0),
        },
        "paper_targets": PAPER_TARGETS,
        "paper_dimensionless_targets": {
            "mass_flow_hat": PAPER_TARGETS["mass_flow_g_s"] * 1.0e-3 / MASS_SCALE,
            "thrust_hat": PAPER_TARGETS["thrust_n"] / THRUST_SCALE,
            "core_exhaust_velocity_hat": PAPER_TARGETS["core_exhaust_velocity_m_s"] / V_PRESSURE,
            "core_mach": PAPER_TARGETS["core_mach"],
            "isentropic_M1": isentropic_ratios(1.0),
            "isentropic_M2p82": isentropic_ratios(2.82),
        },
        "results": results,
    }, indent=2))
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
