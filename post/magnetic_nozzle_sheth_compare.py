"""Compare the Sheth magnetic-nozzle snapshots in SI and paper units.

The solver stores the axial coordinate in x and the two transverse
coordinates in y/z. Julia HDF5 arrays are reversed when read by h5py, so
the reader converts mesh and cell fields to (axial, radial, azimuthal).
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from scipy.special import lambertw


M_H = 1.6735575e-27
N0 = 1.0e18
R_G = 1.380649e-23 / M_H
T0 = 300.0 * 11604.51812
RHO0 = 0.2 * M_H * N0
RHO_REF = 4.0 * M_H * N0 + RHO0
MU0 = 4.0 * np.pi * 1.0e-7
B_STAR = 0.50
ALPHA = 0.50
B_REF = (1.0 + ALPHA) * B_STAR
C_S = np.sqrt(R_G * T0)
# Sheth et al. define V_A with the proton mass density m_H*n_0.
# rho_ref is a separate density normalization and includes the return-ion
# contribution (4*m_H*n_0 + rho_0).
RHO_ALFVEN = M_H * N0
V_A = B_REF / np.sqrt(MU0 * RHO_ALFVEN)
TAU_A = 1.0 / V_A


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
        # Julia's column-major HDF5 layout appears reversed in h5py.
        nodes = [np.asarray(handle[name]).transpose(2, 0, 1)
                 for name in ("x", "y", "z")]
    return tuple(cell_centers(values) for values in nodes)


def read_field(path: Path, name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[name]).transpose(2, 0, 1)


def theory_mach(x: float) -> float:
    b_hat = 1.0 - ALPHA * np.cos(2.0 * np.pi * x)
    b_hat /= 1.0 + ALPHA
    branch = 0 if x < 0.5 else -1
    value = lambertw(-b_hat * b_hat / np.e, branch).real
    return float(np.sqrt(max(0.0, -value)))


def discover_steps(plt_dir: Path, mesh_dir: Path, blocks: int) -> list[int]:
    with h5py.File(mesh_dir / "mesh_b0.h5", "r") as handle:
        mesh_shape = tuple(np.asarray(handle["x"]).shape)
    expected_field_shape = tuple(size - 1 for size in mesh_shape)
    steps = []
    for path in plt_dir.glob("plt-*-b0.h5"):
        try:
            step = int(path.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        complete = all((plt_dir / f"plt-{step}-b{bid}.h5").is_file()
                       for bid in range(blocks))
        if complete:
            with h5py.File(plt_dir / f"plt-{step}-b0.h5", "r") as handle:
                complete = tuple(handle["rho"].shape) == expected_field_shape
        if complete:
            steps.append(step)
    return sorted(set(steps))


def snapshot_time(plt_dir: Path, step: int) -> float:
    root = ET.parse(plt_dir / f"plt-{step}.xmf").getroot()
    time_node = root.find(".//Time")
    if time_node is None or "Value" not in time_node.attrib:
        raise RuntimeError(f"Missing XDMF time for step {step}")
    return float(time_node.attrib["Value"])


def core_profile(
    mesh_dir: Path, plt_dir: Path, step: int, time_over_tau_a: float,
) -> dict[str, np.ndarray]:
    x_nodes, y_nodes, z_nodes = read_mesh(mesh_dir / "mesh_b0.h5")
    radius = np.hypot(y_nodes, z_nodes)
    nx, ny, nz = radius.shape
    flat = np.argmin(radius.reshape(nx, -1), axis=1)
    j = flat // nz
    k = flat % nz
    ii = np.arange(nx)

    def sample(name: str) -> np.ndarray:
        return read_field(plt_dir / f"plt-{step}-b0.h5", name)[ii, j, k]

    x = x_nodes[:, 0, 0]
    rho = sample("rho")
    u = sample("u")
    v = sample("v")
    w = sample("w")
    bx = sample("Bx")
    by = sample("By")
    bz = sample("Bz")
    temperature = sample("T")
    speed = np.sqrt(u * u + v * v + w * w)
    bmag = np.sqrt(bx * bx + by * by + bz * bz)
    mach = speed / C_S
    rows = []
    for index in range(nx):
        rows.append({
            "x_over_L": float(x[index]),
            "time_over_tau_A": float(time_over_tau_a),
            "rho_hat": float(rho[index] / RHO_REF),
            "velocity_hat": float(speed[index] / V_A),
            "mach_sim": float(mach[index]),
            "mach_theory": theory_mach(float(x[index])),
            "B_hat": float(bmag[index] / B_REF),
            "temperature_K": float(temperature[index]),
        })
    return {
        "x": x,
        "rho_hat": rho / RHO_REF,
        "velocity_hat": speed / V_A,
        "mach_sim": mach,
        "mach_theory": np.asarray([row["mach_theory"] for row in rows]),
        "B_hat": bmag / B_REF,
        "temperature_K": temperature,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("mesh_dir", type=Path)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=Path("post/magnetic_nozzle_sheth_results"))
    args = parser.parse_args()
    plt_dir = args.run_dir / "PLT"
    steps = discover_steps(plt_dir, args.mesh_dir, args.blocks)
    if not steps:
        raise RuntimeError(f"No complete five-block snapshots found in {plt_dir}")
    args.output.mkdir(parents=True, exist_ok=True)

    summary = {
        "references": {
            "rho0_kg_m3": RHO0,
            "rho_ref_kg_m3": RHO_REF,
            "rho_alfven_kg_m3": RHO_ALFVEN,
            "T0_K": T0,
            "B_ref_T": B_REF,
            "sound_speed_m_s": C_S,
            "alfven_speed_m_s": V_A,
            "alfven_time_s": TAU_A,
        },
        "steps": steps,
        "snapshots": {},
    }
    for step in steps:
        time_s = snapshot_time(plt_dir, step)
        time_over_tau_a = time_s / TAU_A
        profile = core_profile(
            args.mesh_dir, plt_dir, step, time_over_tau_a,
        )
        mach_error = np.abs(profile["mach_sim"] - profile["mach_theory"])
        snapshot = {
            "step": step,
            "time_s": float(time_s),
            "time_over_tau_A": float(time_over_tau_a),
            "mach_sim_max": float(np.max(profile["mach_sim"])),
            "mach_theory_max": float(np.max(profile["mach_theory"])),
            "mach_linf_error": float(np.max(mach_error)),
            "velocity_hat_max": float(np.max(profile["velocity_hat"])),
            "B_hat_min": float(np.min(profile["B_hat"])),
            "B_hat_max": float(np.max(profile["B_hat"])),
            "rho_hat_min": float(np.min(profile["rho_hat"])),
            "rho_hat_max": float(np.max(profile["rho_hat"])),
            "temperature_min_K": float(np.min(profile["temperature_K"])),
            "temperature_max_K": float(np.max(profile["temperature_K"])),
        }
        summary["snapshots"][str(step)] = snapshot
        rows = profile["rows"]
        with (args.output / f"step_{step}_core_profile.csv").open(
            "w", newline="", encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"step={step} t/tau_A={snapshot['time_over_tau_A']:.8g} "
            f"M_sim_max={snapshot['mach_sim_max']:.8g} "
            f"M_theory_max={snapshot['mach_theory_max']:.8g} "
            f"M_Linf={snapshot['mach_linf_error']:.8g} "
            f"Vhat_max={snapshot['velocity_hat_max']:.8g} "
            f"Bhat=[{snapshot['B_hat_min']:.8g},{snapshot['B_hat_max']:.8g}] "
            f"rhohat=[{snapshot['rho_hat_min']:.8g},{snapshot['rho_hat_max']:.8g}] "
            f"T=[{snapshot['temperature_min_K']:.8g},{snapshot['temperature_max_K']:.8g}]",
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print("references=" + json.dumps(summary["references"], sort_keys=True))


if __name__ == "__main__":
    main()
