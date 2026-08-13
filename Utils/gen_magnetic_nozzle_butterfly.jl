# Five-block butterfly mesh for the magnetic-nozzle case.
#
# Block 0 covers the axis with a Cartesian-like mapped core. Blocks 1--4
# surround it and terminate on the physical circular wall. This removes the
# artificial inner cylindrical wall used by the old six-sector mesh while
# preserving the structured multi-block/CT connectivity contract.

using HDF5
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "core", "structured_interface_transform.jl"))
include(joinpath(@__DIR__, "..", "src", "core", "boundary_types.jl"))

const NG = 4
const paper = "--paper" in ARGS
const sheth_case = "--sheth" in ARGS
const out_dir = get(
    ENV, "MN_MESH_DIR",
    sheth_case ?
        (paper ? "MESH_MAGNETIC_NOZZLE_SHETH_BUTTERFLY" :
                 "MESH_MAGNETIC_NOZZLE_SHETH_BUTTERFLY_SMALL") :
        (paper ? "MESH_MAGNETIC_NOZZLE_BUTTERFLY" :
                 "MESH_MAGNETIC_NOZZLE_BUTTERFLY_SMALL"),
)

const L = 1.0
const radius = 0.20
const core_radius = 0.04
const core_warp = 0.12
# Five blocks with 16 x 16 transverse cells give 1280 transverse cells,
# close to the paper's 32 x 36 = 1152 cells while retaining a nonsingular
# Cartesian core.
const Nx = paper ? 130 : parse(Int, get(ENV, "MN_NX", "24"))
const Ncore = paper ? 16 : parse(Int, get(ENV, "MN_NCORE", "12"))
const Nrad = paper ? 16 : parse(Int, get(ENV, "MN_NRAD", "12"))
const nblocks = 5

const rho0 = sheth_case ? 0.2 * 1.6735575e-27 * 1.0e18 : 5.0e-5
const T0 = sheth_case ? 300.0 * 11604.51812 : 100.0 * 11604.51812
const Bstar = sheth_case ? 0.50 : 0.944
const alpha = 0.50
const kappa = 2.60
const v0 = 0.21

@inline function core_map(s, t)
    return (
        core_radius * s * (1.0 + core_warp * (1.0 - t*t)),
        core_radius * t * (1.0 + core_warp * (1.0 - s*s)),
    )
end

@inline function circle_point(theta)
    return radius * sin(theta), radius * cos(theta)
end

function allocate_block(ny, nz)
    x = zeros(Float64, Nx + 1, ny + 1, nz + 1)
    y = similar(x)
    z = similar(x)
    return x, y, z
end

function fill_center!()
    x, y, z = allocate_block(Ncore, Ncore)
    for k in 1:(Ncore + 1), j in 1:(Ncore + 1), i in 1:(Nx + 1)
        s = 2.0 * (j - 1) / Ncore - 1.0
        t = 2.0 * (k - 1) / Ncore - 1.0
        yc, zc = core_map(s, t)
        x[i, j, k] = L * (i - 1) / Nx
        y[i, j, k] = yc
        z[i, j, k] = zc
    end
    return x, y, z, Nx, Ncore, Ncore
end

function fill_outer!(side)
    ny = side <= 2 ? Nrad : Ncore
    nz = side <= 2 ? Ncore : Nrad
    x, y, z = allocate_block(ny, nz)
    for k in 1:(nz + 1), j in 1:(ny + 1), i in 1:(Nx + 1)
        x[i, j, k] = L * (i - 1) / Nx
        if side == 1 || side == 2
            q = (j - 1) / Nrad
            t = 2.0 * (k - 1) / Ncore - 1.0
            yc, zc = core_map(side == 1 ? -1.0 : 1.0, t)
            # q controls the radial interpolation only. The circular wall
            # must vary along k; reusing q here collapses the outer face.
            theta_fraction = (k - 1) / Ncore
            theta = side == 1 ?
                5.0*pi/4.0 + theta_fraction*(pi/2.0) :
                3.0*pi/4.0 - theta_fraction*(pi/2.0)
            yo, zo = circle_point(theta)
            w_inner = side == 1 ? q : 1.0 - q
        else
            q = (k - 1) / Nrad
            s = 2.0 * (j - 1) / Ncore - 1.0
            yc, zc = core_map(s, side == 3 ? -1.0 : 1.0)
            # q controls the radial interpolation only. The circular wall
            # must vary along j; use the tangential coordinate for theta.
            theta_fraction = (j - 1) / Ncore
            theta = side == 3 ?
                5.0*pi/4.0 - theta_fraction*(pi/2.0) :
                7.0*pi/4.0 + theta_fraction*(pi/2.0)
            yo, zo = circle_point(theta)
            w_inner = side == 3 ? q : 1.0 - q
        end
        y[i, j, k] = w_inner*yc + (1.0 - w_inner)*yo
        z[i, j, k] = w_inner*zc + (1.0 - w_inner)*zo
    end
    return x, y, z, Nx, ny, nz
end

blocks = [fill_center!(), fill_outer!(1), fill_outer!(2), fill_outer!(3), fill_outer!(4)]
mkpath(out_dir)

for bid in 0:4
    x, y, z, nx, ny, nz = blocks[bid + 1]
    h5open(joinpath(out_dir, "mesh_b$(bid).h5"), "w") do file
        file["NG"] = Int32(NG)
        file["Nx"] = Int32(nx)
        file["Ny"] = Int32(ny)
        file["Nz"] = Int32(nz)
        file["coords"] = cat(
            reshape(x, 1, size(x)...), reshape(y, 1, size(y)...),
            reshape(z, 1, size(z)...); dims=1,
        )
        file["x"] = x
        file["y"] = y
        file["z"] = z
    end
end

connections = Int32[
    0 3 1 4; 1 4 0 3; 0 4 2 3; 2 3 0 4;
    0 5 3 6; 3 6 0 5; 0 6 4 5; 4 5 0 6;
    1 5 3 3; 3 3 1 5; 1 6 4 3; 4 3 1 6;
    2 5 3 4; 3 4 2 5; 2 6 4 4; 4 4 2 6
]

function tangent(bid, face)
    x, y, z, nx, ny, nz = blocks[bid + 1]
    ii = (nx + 2) ÷ 2
    if face == 3 || face == 4
        jj = face == 3 ? 1 : ny + 1
        kk = (nz + 2) ÷ 2
        return [y[ii, jj, kk + 1] - y[ii, jj, kk], z[ii, jj, kk + 1] - z[ii, jj, kk]]
    end
    kk = face == 5 ? 1 : nz + 1
    jj = (ny + 2) ÷ 2
    return [y[ii, jj + 1, kk] - y[ii, jj, kk], z[ii, jj + 1, kk] - z[ii, jj, kk]]
end

reverse_tan = zeros(Int32, size(connections, 1))
for row in 1:size(connections, 1)
    b1, f1, b2, f2 = connections[row, :]
    reverse_tan[row] = dot(tangent(b1, f1), tangent(b2, f2)) < 0 ? 1 : 0
end
flip_normal = zeros(Int32, size(connections, 1))
for row in axes(connections, 1)
    _, f1, _, f2 = connections[row, :]
    # A face-normal component changes local axis when eta and zeta faces
    # meet. The halo exchange validator requires this explicit flag.
    flip_normal[row] = Int32(
        structured_face_normal_axis(f1) != structured_face_normal_axis(f2),
    )
end
axis_map = structured_connectivity_axis_map(connections, reverse_tan)

face_bc = fill(Int32(0), nblocks, 6)
for bid in 1:nblocks
    face_bc[bid, 1] = sheth_case ? BC_MHD_PROFILED_INFLOW : BC_MHD_RESERVOIR_INFLOW
    face_bc[bid, 2] = BC_MHD_OUTFLOW_EXTERNAL_FIELD
end
# The physical circular wall is face 3/4/5/6 for blocks 1/2/3/4.
for (bid, face) in ((2, 3), (3, 4), (4, 5), (5, 6))
    face_bc[bid, face] = sheth_case ? BC_MHD_FIXED_EXTERNAL_FIELD : BC_MHD_EXTERNAL_FIELD
end

bc_params = zeros(Float64, nblocks, 6, 20)
for bid in 1:nblocks, face in (1, 2, 3, 4, 5, 6)
    bcp = view(bc_params, bid, face, :)
    bcp[BCP_MN_RHO0] = rho0
    bcp[BCP_MN_T0] = T0
    bcp[BCP_MN_B0] = Bstar
    bcp[BCP_MN_XC] = sheth_case ? 0.0 : 0.50
    bcp[BCP_MN_RB] = 0.20
    bcp[BCP_MN_LB] = sheth_case ? L : 0.32
    bcp[BCP_MN_TURNS] = 32.0
    bcp[BCP_MN_MODEL] = sheth_case ? 1.0 : 0.0
    bcp[BCP_MN_ALPHA] = alpha
    bcp[BCP_MN_KAPPA] = kappa
    bcp[BCP_MN_V0] = v0
end

h5open(joinpath(out_dir, "block_connectivity.h5"), "w") do file
    file["Nblocks"] = Int32(nblocks)
    file["Nx_b"] = Int32[Nx for _ in 1:nblocks]
    file["Ny_b"] = Int32[Ncore, Nrad, Nrad, Ncore, Ncore]
    file["Nz_b"] = Int32[Ncore, Ncore, Ncore, Nrad, Nrad]
    file["connectivity"] = connections
    file["reverse_tan"] = reverse_tan
    file["flip_normal"] = flip_normal
    file["axis_map"] = axis_map
    file["face_bc"] = face_bc
    file["bc_params"] = bc_params
end

println("MAGNETIC_NOZZLE_BUTTERFLY dir=$(out_dir) blocks=$(nblocks) " *
        "grid=$(Nx)x$(Ncore)x$(Ncore) core=$(core_radius) radius=$(radius)")
