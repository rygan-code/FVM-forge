# gen_pipe_ac_connectivity.jl — Generate AC-compatible block_connectivity.h5
# 
# Reads the existing compressible pipe connectivity and creates a proper
# AC version with:
#   - wall (1) → ac_wall (20)
#   - bc_params expanded to N_BC_PARAMS = 17 columns
#
# Usage: julia gen_pipe_ac_connectivity.jl <mesh_dir>
# Example: julia gen_pipe_ac_connectivity.jl MESH_coarse
#          julia gen_pipe_ac_connectivity.jl MESH

using HDF5

# BC constants (from bc_types.jl)
const BC_ISOTHERMAL_WALL = Int32(1)
const BC_AC_WALL         = Int32(20)
const N_BC_PARAMS        = 17

mesh_dir = length(ARGS) >= 1 ? ARGS[1] : "MESH"
src_path = joinpath(mesh_dir, "block_connectivity.h5")
dst_path = joinpath(mesh_dir, "block_connectivity_ac.h5")

if !isfile(src_path)
    error("Source connectivity not found: $src_path")
end

println(">>> Reading: $src_path")

# Read all datasets from original
Nblocks, Nx_b, Ny_b, Nz_b = h5open(src_path, "r") do f
    (Int(read(f["Nblocks"])), read(f["Nx_b"]), read(f["Ny_b"]), read(f["Nz_b"]))
end

connectivity = h5read(src_path, "connectivity")
face_bc      = h5read(src_path, "face_bc")
bc_params_old = h5read(src_path, "bc_params")
flip_normal  = h5read(src_path, "flip_normal")
reverse_tan  = h5read(src_path, "reverse_tan")

println("  Nblocks = $Nblocks")
println("  Grid per block: $(Nx_b[1])×$(Ny_b[1])×$(Nz_b[1])")
println("  bc_params shape: $(size(bc_params_old)) → will expand to ($Nblocks, 6, $N_BC_PARAMS)")

# Patch face_bc: wall(1) → ac_wall(20)
n_patched = count(==(BC_ISOTHERMAL_WALL), face_bc)
face_bc[face_bc .== BC_ISOTHERMAL_WALL] .= BC_AC_WALL
println("  Patched $n_patched wall faces → ac_wall")

# Print final BC layout
bc_id_to_name = Dict(
    0 => "interblock", 1 => "wall", 2 => "periodic",
    20 => "ac_wall", 21 => "ac_lid"
)
face_names = ["ξ-", "ξ+", "η-", "η+", "ζ-", "ζ+"]
for bid in 1:Nblocks
    bcs = join(["$(face_names[f])=$(get(bc_id_to_name, Int(face_bc[bid,f]), "?"))" for f in 1:6], "  ")
    println("  Block $(bid-1): $bcs")
end

# Expand bc_params to N_BC_PARAMS
nblk, nfaces, nparams_old = size(bc_params_old)
bc_params_new = zeros(FT, nblk, nfaces, N_BC_PARAMS)
bc_params_new[:, :, 1:nparams_old] .= bc_params_old

# Write output
isfile(dst_path) && rm(dst_path)
h5open(dst_path, "w") do f
    f["Nblocks"]     = Int32(Nblocks)
    f["Nx_b"]        = Int32.(Nx_b)
    f["Ny_b"]        = Int32.(Ny_b)
    f["Nz_b"]        = Int32.(Nz_b)
    f["connectivity"] = connectivity
    f["face_bc"]     = face_bc
    f["bc_params"]   = bc_params_new
    f["flip_normal"] = flip_normal
    f["reverse_tan"] = reverse_tan
end

println()
println("✓ AC connectivity written to: $dst_path")
println("  bc_params: ($nblk, $nfaces, $nparams_old) → ($nblk, $nfaces, $N_BC_PARAMS)")
