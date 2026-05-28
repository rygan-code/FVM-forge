# analyze_interface_mismatch.jl — Check grid spacing at block interfaces
# Run: julia Utils/analyze_interface_mismatch.jl [MESH_DIR]
using HDF5, Printf, LinearAlgebra

mesh_dir = length(ARGS) >= 1 ? ARGS[1] : "MESH"
println("═" ^ 70)
println("  Block Interface Grid Spacing Analysis")
println("  Mesh: $mesh_dir")
println("═" ^ 70)

# Load all 5 blocks
blocks = []
for bid in 0:4
    mf = joinpath(mesh_dir, "mesh_b$bid.h5")
    h5 = h5open(mf, "r")
    Nx = read(h5["Nx"]); Ny = read(h5["Ny"]); Nz = read(h5["Nz"])
    y = read(h5["y"]); z = read(h5["z"])
    close(h5)
    push!(blocks, (Nx=Nx, Ny=Ny, Nz=Nz, y=y, z=z))
end

# Helper: compute local cell size (in y-z plane) at a face
function face_cell_sizes(blk, face_id)
    y, z = blk.y, blk.z
    Nx, Ny, Nz = blk.Nx, blk.Ny, blk.Nz
    mid_i = (Nx + 2) ÷ 2  # mid-axial slice
    
    if face_id == 3  # η- (j=1)
        # Normal direction is j, compute Δ in j-direction at j=1
        sizes = Float64[]
        for k in 1:Nz+1
            dy = y[mid_i, 2, k] - y[mid_i, 1, k]
            dz = z[mid_i, 2, k] - z[mid_i, 1, k]
            push!(sizes, sqrt(dy^2 + dz^2))
        end
        return sizes
    elseif face_id == 4  # η+ (j=Ny+1)
        sizes = Float64[]
        for k in 1:Nz+1
            dy = y[mid_i, Ny+1, k] - y[mid_i, Ny, k]
            dz = z[mid_i, Ny+1, k] - z[mid_i, Ny, k]
            push!(sizes, sqrt(dy^2 + dz^2))
        end
        return sizes
    elseif face_id == 5  # ζ- (k=1)
        sizes = Float64[]
        for j in 1:Ny+1
            dy = y[mid_i, j, 2] - y[mid_i, j, 1]
            dz = z[mid_i, j, 2] - z[mid_i, j, 1]
            push!(sizes, sqrt(dy^2 + dz^2))
        end
        return sizes
    elseif face_id == 6  # ζ+ (k=Nz+1)
        sizes = Float64[]
        for j in 1:Ny+1
            dy = y[mid_i, j, Nz+1] - y[mid_i, j, Nz]
            dz = z[mid_i, j, Nz+1] - z[mid_i, j, Nz]
            push!(sizes, sqrt(dy^2 + dz^2))
        end
        return sizes
    end
end

# Block connectivity (from gen_butterfly_fvm.jl)
# Format: (block_A, face_A) <-> (block_B, face_B)
interfaces = [
    (0, 3, 1, 4),  # Block0 η- ↔ Block1 η+
    (0, 4, 2, 3),  # Block0 η+ ↔ Block2 η-
    (0, 5, 3, 6),  # Block0 ζ- ↔ Block3 ζ+
    (0, 6, 4, 5),  # Block0 ζ+ ↔ Block4 ζ-
    (1, 5, 3, 3),  # Block1 ζ- ↔ Block3 η-
    (1, 6, 4, 3),  # Block1 ζ+ ↔ Block4 η-
    (2, 5, 3, 4),  # Block2 ζ- ↔ Block3 η+
    (2, 6, 4, 4),  # Block2 ζ+ ↔ Block4 η+
]

face_names = Dict(3 => "η-", 4 => "η+", 5 => "ζ-", 6 => "ζ+")

println()
println("  ┌──────────────────────────────┬──────────────────────────────┬────────────┐")
println("  │        Side A                │        Side B                │  Mismatch  │")
println("  │  Block  Face   Δ_min   Δ_max │  Block  Face   Δ_min   Δ_max│   Ratio    │")
println("  ├──────────────────────────────┼──────────────────────────────┼────────────┤")

for (bA, fA, bB, fB) in interfaces
    sA = face_cell_sizes(blocks[bA+1], fA)
    sB = face_cell_sizes(blocks[bB+1], fB)
    
    minA, maxA = minimum(sA), maximum(sA)
    minB, maxB = minimum(sB), maximum(sB)
    
    # Compute point-wise mismatch ratio
    # Sizes may have different lengths if blocks have different N along that face
    # Use representative stats
    meanA = sum(sA) / length(sA)
    meanB = sum(sB) / length(sB)
    ratio = max(meanA, meanB) / min(meanA, meanB)
    
    @printf("  │  B%d   %s  %6.4f  %6.4f │  B%d   %s  %6.4f  %6.4f│   %5.2f    │\n",
            bA, face_names[fA], minA*1000, maxA*1000,
            bB, face_names[fB], minB*1000, maxB*1000,
            ratio)
end
println("  └──────────────────────────────┴──────────────────────────────┴────────────┘")
println("  (Δ values in mm)")

# ── Detailed analysis of center-annular interface ──
println()
println("  ═══ Center Block (B0) ↔ Annular Block Interface Detail ═══")
b0 = blocks[1]
mid_i = (b0.Nx + 2) ÷ 2

# Block 0 cell sizes in j and k directions
dy_b0_j = [sqrt((b0.y[mid_i,j+1,b0.Nz÷2+1]-b0.y[mid_i,j,b0.Nz÷2+1])^2 + 
                 (b0.z[mid_i,j+1,b0.Nz÷2+1]-b0.z[mid_i,j,b0.Nz÷2+1])^2) 
           for j in 1:b0.Ny]
dy_b0_k = [sqrt((b0.y[mid_i,b0.Ny÷2+1,k+1]-b0.y[mid_i,b0.Ny÷2+1,k])^2 + 
                 (b0.z[mid_i,b0.Ny÷2+1,k+1]-b0.z[mid_i,b0.Ny÷2+1,k])^2) 
           for k in 1:b0.Nz]

# Block 1 cell sizes at inner boundary (η+ face, j=Ny+1)
b1 = blocks[2]
mid_i1 = (b1.Nx + 2) ÷ 2
# Last cell in j-direction (inner boundary)
dy_b1_inner = sqrt((b1.y[mid_i1,b1.Ny+1,b1.Nz÷2+1]-b1.y[mid_i1,b1.Ny,b1.Nz÷2+1])^2 + 
                    (b1.z[mid_i1,b1.Ny+1,b1.Nz÷2+1]-b1.z[mid_i1,b1.Ny,b1.Nz÷2+1])^2)
# First cell (wall)
dy_b1_wall = sqrt((b1.y[mid_i1,2,b1.Nz÷2+1]-b1.y[mid_i1,1,b1.Nz÷2+1])^2 + 
                   (b1.z[mid_i1,2,b1.Nz÷2+1]-b1.z[mid_i1,1,b1.Nz÷2+1])^2)

@printf("\n  Block 0 (center):\n")
@printf("    j-direction cell: min=%.4f, max=%.4f, mean=%.4f mm\n", 
        minimum(dy_b0_j)*1000, maximum(dy_b0_j)*1000, sum(dy_b0_j)/length(dy_b0_j)*1000)
@printf("    k-direction cell: min=%.4f, max=%.4f, mean=%.4f mm\n", 
        minimum(dy_b0_k)*1000, maximum(dy_b0_k)*1000, sum(dy_b0_k)/length(dy_b0_k)*1000)
@printf("\n  Block 1 (annular):\n")
@printf("    Wall cell (j=1):    %.4f mm\n", dy_b1_wall*1000)
@printf("    Inner cell (j=N):   %.4f mm\n", dy_b1_inner*1000)
@printf("    Inner/Center ratio: %.2f\n", dy_b1_inner / (sum(dy_b0_j)/length(dy_b0_j)))

println()
println("═" ^ 70)
