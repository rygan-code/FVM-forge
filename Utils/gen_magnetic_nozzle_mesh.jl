# Generate the structured mesh for the Lorzel-Mikellides magnetic nozzle.
#
# The logical topology is 6 azimuthal sectors with the full axial direction
# inside each block. Each block contains (nax, nrad, ntheta) cells, giving the
# paper's 130 x 32 x 36 physical grid when run with --paper. The 1 cm inner
# radius removes the cylindrical-axis singularity used in the paper.

using HDF5
using Printf

if !isdefined(@__MODULE__, :StructuredFaceFrame)
    include(joinpath(@__DIR__, "..", "src", "core", "structured_interface_transform.jl"))
end
if !isdefined(@__MODULE__, :BC_MHD_PROFILED_INFLOW)
    include(joinpath(@__DIR__, "..", "src", "core", "boundary_types.jl"))
end

const NG = 4
const small_case = !("--paper" in ARGS)
const sheth_case = "--sheth" in ARGS
const out_dir = get(
    ENV, "MN_MESH_DIR",
    sheth_case ?
        (small_case ? "MESH_MAGNETIC_NOZZLE_SHETH_SMALL" : "MESH_MAGNETIC_NOZZLE_SHETH") :
        (small_case ? "MESH_MAGNETIC_NOZZLE_SMALL" : "MESH_MAGNETIC_NOZZLE"),
)
const nrad = small_case ? 8 : 32
# Every real direction must contain at least NG cells because interblock
# coordinate ghosts are copied from the neighbor's real-only node array.
const ntheta = small_case ? 6 : 6
const nax = sheth_case ? (small_case ? 24 : 256) : (small_case ? 6 : 130)
const nrad_effective = sheth_case ? (small_case ? 16 : 156) : nrad
const nsector = 6
# CT multi-block synchronization currently supports transverse interfaces
# (faces 3:6). Keep the full axial direction inside each sector block so the
# paper-sized 130 x 32 x 36 physical grid remains unchanged.
const n_axial_blocks = 1
const nblocks = nsector * n_axial_blocks

const length_domain = 1.0
const radius_inner = sheth_case ? 0.01 : 0.01
const radius_outer = sheth_case ? 0.20 : 0.18
const coil_center = 0.50
const coil_radius = 0.20
const coil_length = 0.32
const coil_turns = 32
const rho0 = 5.0e-5
const T0 = 100.0 * 11604.51812
const p0 = 0.355e6
# Connectivity boundary parameters use the same SI Tesla convention as the
# solver's MHD state and the FiniteSolenoidMagneticField model.
const B0_Tesla = 0.944
const sheth_rho0 = 0.2 * 1.6735575e-27 * 1.0e18
const sheth_T0 = (300.0 * 11604.51812) / (5.0/3.0)
const sheth_Bstar = 0.50
const sheth_alpha = 0.50
const sheth_kappa = 2.60
const sheth_v0 = 0.21

function block_id(axial_block, sector)
    return axial_block * nsector + sector
end

function build_block(axial_block, sector)
    x0 = length_domain * axial_block / n_axial_blocks
    x1 = length_domain * (axial_block + 1) / n_axial_blocks
    theta0 = 2.0 * pi * sector / nsector
    theta1 = 2.0 * pi * (sector + 1) / nsector
    radial = collect(range(radius_inner, radius_outer; length=nrad_effective + 1))
    theta = collect(range(theta0, theta1; length=ntheta + 1))
    axial = collect(range(x0, x1; length=nax + 1))

    x = zeros(Float64, nax + 1, nrad_effective + 1, ntheta + 1)
    y = similar(x)
    z = similar(x)
    for k in axes(x, 3), j in axes(x, 2), i in axes(x, 1)
        r = radial[j]
        angle = theta[k]
        x[i, j, k] = axial[i]
        y[i, j, k] = r * cos(angle)
        z[i, j, k] = r * sin(angle)
    end
    return x, y, z
end

function add_connection!(rows, block_a, face_a, block_b, face_b)
    push!(rows, Int32[block_a, face_a, block_b, face_b])
    push!(rows, Int32[block_b, face_b, block_a, face_a])
end

mkpath(out_dir)
for axial_block in 0:(n_axial_blocks - 1), sector in 0:(nsector - 1)
    bid = block_id(axial_block, sector)
    x, y, z = build_block(axial_block, sector)
    h5open(joinpath(out_dir, "mesh_b$(bid).h5"), "w") do file
        file["NG"] = Int32(NG)
        file["Nx"] = Int32(nax)
    file["Ny"] = Int32(nrad_effective)
        file["Nz"] = Int32(ntheta)
        file["coords"] = cat(
            reshape(x, 1, size(x)...), reshape(y, 1, size(y)...),
            reshape(z, 1, size(z)...); dims=1,
        )
        file["x"] = x
        file["y"] = y
        file["z"] = z
    end
end

rows = Vector{Vector{Int32}}()
# Axial block interfaces: face 2 (+x) to face 1 (-x).
for axial_block in 0:(n_axial_blocks - 2), sector in 0:(nsector - 1)
    add_connection!(
        rows, block_id(axial_block, sector), 2,
        block_id(axial_block + 1, sector), 1,
    )
end
# Azimuthal interfaces, including the periodic wrap from sector 5 to 0.
for axial_block in 0:(n_axial_blocks - 1), sector in 0:(nsector - 1)
    next_sector = mod(sector + 1, nsector)
    add_connection!(
        rows, block_id(axial_block, sector), 6,
        block_id(axial_block, next_sector), 5,
    )
end
connectivity = reduce(vcat, (reshape(row, 1, 4) for row in rows))

face_bc = fill(Int32(0), nblocks, 6)
for axial_block in 0:(n_axial_blocks - 1), sector in 0:(nsector - 1)
    bid = block_id(axial_block, sector) + 1
    face_bc[bid, 1] = axial_block == 0 ?
        (sheth_case ? BC_MHD_PROFILED_INFLOW : BC_MHD_RESERVOIR_INFLOW) : Int32(0)
    face_bc[bid, 2] = axial_block == n_axial_blocks - 1 ?
        (sheth_case ? BC_MHD_OUTFLOW_EXTERNAL_FIELD : BC_ZERO_GRADIENT) : Int32(0)
    face_bc[bid, 3] = BC_SLIP_WALL  # symmetry approximation at the small inner cutout
    face_bc[bid, 4] = sheth_case ? BC_MHD_FIXED_EXTERNAL_FIELD : BC_MHD_EXTERNAL_FIELD
end

bc_params = zeros(Float64, nblocks, 6, 20)
for axial_block in 0:(n_axial_blocks - 1), sector in 0:(nsector - 1)
    bid = block_id(axial_block, sector) + 1
    inlet = view(bc_params, bid, 1, :)
    if sheth_case
        inlet[1] = sheth_rho0
        inlet[2] = sheth_T0
        inlet[3] = sheth_Bstar
        inlet[4] = 0.0
        inlet[5] = radius_outer
        inlet[6] = length_domain
        inlet[17] = 1.0
        inlet[18] = sheth_alpha
        inlet[19] = sheth_kappa
        inlet[20] = sheth_v0
    else
        inlet[1] = rho0
        inlet[2] = T0
        inlet[3] = B0_Tesla
        inlet[4] = coil_center
        inlet[5] = coil_radius
        inlet[6] = coil_length
        inlet[7] = coil_turns
    end

    outer = view(bc_params, bid, 4, :)
    if sheth_case
        outer[1] = sheth_rho0
        outer[2] = sheth_T0
        outer[3] = sheth_Bstar
        outer[4] = 0.0
        outer[5] = radius_outer
        outer[6] = length_domain
        outer[17] = 1.0
        outer[18] = sheth_alpha
        outer[19] = sheth_kappa
        outer[20] = sheth_v0
    else
        outer[3] = B0_Tesla
        outer[4] = coil_center
        outer[5] = coil_radius
        outer[6] = coil_length
        outer[7] = coil_turns
    end

    if sheth_case
        # The outlet fluid is extrapolated, but the paper's background field
        # remains fixed on every physical boundary sheet.
        outlet = view(bc_params, bid, 2, :)
        outlet .= inlet
    end
end

h5open(joinpath(out_dir, "block_connectivity.h5"), "w") do file
    file["Nblocks"] = Int32(nblocks)
    file["Nx_b"] = fill(Int32(nax), nblocks)
    file["Ny_b"] = fill(Int32(nrad_effective), nblocks)
    file["Nz_b"] = fill(Int32(ntheta), nblocks)
    file["connectivity"] = connectivity
    reverse_tan = zeros(Int32, size(connectivity, 1))
    file["reverse_tan"] = reverse_tan
    file["flip_normal"] = zeros(Int32, size(connectivity, 1))
    file["axis_map"] = structured_connectivity_axis_map(connectivity, reverse_tan)
    file["face_bc"] = face_bc
    file["bc_params"] = bc_params
end

@printf(
    "MAGNETIC_NOZZLE_MESH dir=%s blocks=%d cells_per_block=%d x %d x %d total_cells=%d mode=%s\n",
    out_dir, nblocks, nax, nrad_effective, ntheta, nblocks*nax*nrad_effective*ntheta,
    sheth_case ? (small_case ? "sheth-small" : "sheth-paper") :
        (small_case ? "small" : "paper"),
)
