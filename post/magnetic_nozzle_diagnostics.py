from __future__ import annotations

import argparse
import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np


GAMMA = 5.0 / 3.0
THROAT_X_M = 0.44
PAPER_TARGETS = {
    "mass_flow_g_s": 55.2,
    "thrust_n": 8650.0,
    "core_mach": 2.80,
    "core_exhaust_velocity_m_s": 157960.0,
    "current_layer_thickness_m": 0.035,
}


@dataclass(frozen=True)
class MeshBlock:
    axial_centers: np.ndarray
    radial_centers: np.ndarray
    cross_section_area: np.ndarray
    storage_permutation: tuple[int, int, int]


@dataclass(frozen=True)
class NozzleDiagnostics:
    step: int
    time_us: float
    mass_flow_g_s: float
    thrust_n: float
    core_mach: float
    core_exhaust_velocity_m_s: float
    core_temperature_k: float
    throat_x_m: float
    current_layer_thickness_m: float
    current_layer_cells: float


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


def cross_section_cell_area(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
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


def load_mesh_block(path: Path) -> MeshBlock:
    with h5py.File(path, "r") as mesh:
        x = np.asarray(mesh["x"], dtype=np.float64)
        y = np.asarray(mesh["y"], dtype=np.float64)
        z = np.asarray(mesh["z"], dtype=np.float64)
    radius_nodes = np.hypot(y, z)
    axial_variation = [
        float(np.ptp(np.mean(x, axis=tuple(other for other in range(3) if other != axis))))
        for axis in range(3)
    ]
    radial_variation = [
        float(np.ptp(np.mean(
            radius_nodes,
            axis=tuple(other for other in range(3) if other != axis),
        )))
        for axis in range(3)
    ]
    axial_axis = int(np.argmax(axial_variation))
    radial_axis = int(np.argmax(radial_variation))
    if axial_axis == radial_axis:
        raise RuntimeError(f"could not distinguish axial and radial axes in {path}")
    azimuthal_axis = next(axis for axis in range(3) if axis not in (axial_axis, radial_axis))
    permutation = (axial_axis, radial_axis, azimuthal_axis)
    x = np.transpose(x, permutation)
    y = np.transpose(y, permutation)
    z = np.transpose(z, permutation)
    xc = cell_centers(x)
    yc = cell_centers(y)
    zc = cell_centers(z)
    radius = np.hypot(yc, zc)
    return MeshBlock(
        axial_centers=np.mean(xc, axis=(1, 2)),
        radial_centers=np.mean(radius, axis=(0, 2)),
        cross_section_area=cross_section_cell_area(x, y, z),
        storage_permutation=permutation,
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def first_crossing(radius: np.ndarray, profile: np.ndarray, level: float) -> float:
    for index in range(len(radius) - 1):
        left = profile[index] - level
        right = profile[index + 1] - level
        if left == 0.0:
            return float(radius[index])
        if left * right <= 0.0 and profile[index + 1] != profile[index]:
            fraction = (level - profile[index]) / (profile[index + 1] - profile[index])
            return float(radius[index] + fraction * (radius[index + 1] - radius[index]))
    return float("nan")


def pressure_layer_thickness(radius: np.ndarray, pressure: np.ndarray) -> float:
    pressure_drop = pressure[0] - pressure[-1]
    scale = max(float(np.max(np.abs(pressure))), np.finfo(float).tiny)
    if pressure_drop <= 1.0e-6 * scale:
        return float("nan")
    normalized = (pressure - pressure[-1]) / pressure_drop
    radius_90 = first_crossing(radius, normalized, 0.9)
    radius_10 = first_crossing(radius, normalized, 0.1)
    return radius_10 - radius_90


def parse_times(log_path: Path) -> dict[int, float]:
    pattern = re.compile(r"^Step:\s*(\d+)\s+Time:\s*([+\-0-9.eE]+)")
    times: dict[int, float] = {}
    if not log_path.is_file():
        return times
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            times[int(match.group(1))] = float(match.group(2)) * 1.0e6
    return times


def discover_steps(plt_dir: Path, blocks: int) -> list[int]:
    pattern = re.compile(r"plt-(\d+)-b0\.h5$")
    candidates = sorted(
        int(match.group(1))
        for path in plt_dir.glob("plt-*-b0.h5")
        if (match := pattern.match(path.name))
    )
    return [
        step
        for step in candidates
        if all((plt_dir / f"plt-{step}-b{block}.h5").is_file() for block in range(blocks))
    ]


def analyze_step(
    plt_dir: Path,
    meshes: list[MeshBlock],
    step: int,
    time_us: float,
) -> NozzleDiagnostics:
    mass_flow = 0.0
    thrust = 0.0
    core_velocity_sum = 0.0
    core_mach_sum = 0.0
    core_temperature_sum = 0.0
    core_weight = 0.0
    throat_pressure_weighted = np.zeros_like(meshes[0].radial_centers)
    throat_area = np.zeros_like(meshes[0].radial_centers)

    throat_index = int(np.argmin(np.abs(meshes[0].axial_centers - THROAT_X_M)))
    for block, mesh in enumerate(meshes):
        with h5py.File(plt_dir / f"plt-{step}-b{block}.h5", "r") as output:
            rho = np.transpose(np.asarray(output["rho"]), mesh.storage_permutation)
            u = np.transpose(np.asarray(output["u"]), mesh.storage_permutation)
            v = np.transpose(np.asarray(output["v"]), mesh.storage_permutation)
            w = np.transpose(np.asarray(output["w"]), mesh.storage_permutation)
            pressure = np.transpose(np.asarray(output["p"]), mesh.storage_permutation)
            temperature = np.transpose(np.asarray(output["T"]), mesh.storage_permutation)

        exit_area = mesh.cross_section_area
        exit_rho = rho[-1]
        exit_u = u[-1]
        mass_flow += float(np.sum(exit_rho * exit_u * exit_area))
        thrust += float(np.sum(exit_rho * exit_u * exit_u * exit_area))

        core_area = exit_area[0]
        core_speed = np.sqrt(u[-1, 0] ** 2 + v[-1, 0] ** 2 + w[-1, 0] ** 2)
        sound_speed = np.sqrt(GAMMA * pressure[-1, 0] / rho[-1, 0])
        core_velocity_sum += float(np.sum(u[-1, 0] * core_area))
        core_mach_sum += float(np.sum((core_speed / sound_speed) * core_area))
        core_temperature_sum += float(np.sum(temperature[-1, 0] * core_area))
        core_weight += float(np.sum(core_area))

        for radial_index in range(len(mesh.radial_centers)):
            weights = exit_area[radial_index]
            throat_pressure_weighted[radial_index] += float(
                np.sum(pressure[throat_index, radial_index] * weights)
            )
            throat_area[radial_index] += float(np.sum(weights))

    throat_pressure = throat_pressure_weighted / throat_area
    layer_thickness = pressure_layer_thickness(
        meshes[0].radial_centers, throat_pressure,
    )
    radial_spacing = float(np.mean(np.diff(meshes[0].radial_centers)))
    return NozzleDiagnostics(
        step=step,
        time_us=time_us,
        mass_flow_g_s=mass_flow * 1.0e3,
        thrust_n=thrust,
        core_mach=core_mach_sum / core_weight,
        core_exhaust_velocity_m_s=core_velocity_sum / core_weight,
        core_temperature_k=core_temperature_sum / core_weight,
        throat_x_m=float(meshes[0].axial_centers[throat_index]),
        current_layer_thickness_m=layer_thickness,
        current_layer_cells=layer_thickness / radial_spacing,
    )


def print_results(results: list[NozzleDiagnostics]) -> None:
    columns = tuple(asdict(results[0]).keys())
    print(",".join(columns))
    for result in results:
        values = asdict(result)
        print(",".join(
            str(values[column]) if column == "step" else f"{values[column]:.9g}"
            for column in columns
        ))

    final = asdict(results[-1])
    print("\nPaper-target relative errors:")
    for name, target in PAPER_TARGETS.items():
        value = final[name]
        relative_error = (value - target) / target
        print(
            f"  {name}: value={value:.9g} target={target:.9g} "
            f"relative_error={relative_error:+.3%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("mesh_dir", type=Path)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    plt_dir = args.run_dir / "PLT" if (args.run_dir / "PLT").is_dir() else args.run_dir
    steps = discover_steps(plt_dir, args.blocks)
    if not steps:
        raise RuntimeError(f"no complete PLT steps found in {plt_dir}")
    meshes = [
        load_mesh_block(args.mesh_dir / f"mesh_b{block}.h5")
        for block in range(args.blocks)
    ]
    times = parse_times(args.run_dir / "run.log")
    results = [
        analyze_step(plt_dir, meshes, step, times.get(step, float("nan")))
        for step in steps
    ]
    print_results(results)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)


if __name__ == "__main__":
    main()
