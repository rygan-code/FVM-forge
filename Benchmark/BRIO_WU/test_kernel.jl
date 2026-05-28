# Minimal test to isolate the CUDA ILLEGAL_ADDRESS in MHD Brio-Wu
# Tests each kernel individually with synchronization after each

# Load the solver (includes all kernel definitions)
include("run.jl")

using CUDA

println("\n=== Kernel Isolation Test ===")
println("Testing each kernel individually with explicit sync...")

# Get the block
b = first(values(blocks))
nxp, nyp, nzp = b.Nx, b.Ny, b.Nz
println("Grid: $(nxp)×$(nyp)×$(nzp), NG=$NG, Ncons=$Ncons, Nprim=$Nprim")
println("Q size: ", size(b.Q))
println("U size: ", size(b.U))
println("ϕ size: ", size(b.ϕ))
println("Areai size: ", size(b.Areai))
println("Areaj size: ", size(b.Areaj))
println("Areak size: ", size(b.Areak))
println("nxi size: ", size(b.nxi))
println("Vol size: ", size(b.Vol))
println("stencil_i size: ", size(b.stencil_i))
println("Δstencil_i size: ", size(b.Δstencil_i))
println("lin_phi_i size: ", size(b.lin_phi_i))

# Check for NaN/Inf in initial state
println("\nQ has NaN: ", any(isnan, Array(b.Q)))
println("U has NaN: ", any(isnan, Array(b.U)))
println("Q has Inf: ", any(isinf, Array(b.Q)))
println("U has Inf: ", any(isinf, Array(b.U)))
println("Q min/max: ", extrema(Array(b.Q)))
println("U min/max: ", extrema(Array(b.U)))

# Test shockSensor
println("\n--- Testing shockSensor ---")
nb = (cld(nxp+2*NG, nthreads[1]), cld(nyp+2*NG, nthreads[2]), cld(nzp+2*NG, nthreads[3]))
@gpu_launch threads=nthreads blocks=nb shockSensor(b.ϕ, b.Q, nxp, nyp, nzp)
CUDA.synchronize()
println("  OK! ϕ min/max: ", extrema(Array(b.ϕ)))

# Test Conser_reconstruct_i
println("\n--- Testing Conser_reconstruct_i ---")
Fx = gpu_zeros(FT, nxp+1, nyp, nzp, Ncons)
println("  Fx size: ", size(Fx))
threads_ri = (Int32(4), Int32(8), Int32(16))
nb_ri = (Int32(cld(nxp+2*NG, threads_ri[1])), Int32(cld(nyp+2*NG, threads_ri[2])), Int32(cld(nzp+2*NG, threads_ri[3])))
println("  Threads: $threads_ri, Blocks: $nb_ri")
println("  Total threads: $(prod(threads_ri .* nb_ri))")
println("  i range: $NG .. $(nxp+NG), j range: $(NG+1) .. $(nyp+NG), k range: $(NG+1) .. $(nzp+NG)")

# Check array index bounds
println("\n  Array bounds checking:")
println("    U[i-3,j,k,n]: i-3 min = $(NG-3) = $(NG-3)")
println("    U[i+4,j,k,n]: i+4 max = $(nxp+NG+4) = $(nxp+NG+4)")
println("    U dim1 = $(size(b.U,1)), need max $(nxp+NG+4)")
println("    U dim2 = $(size(b.U,2)), j max = $(nyp+NG)")
println("    U dim3 = $(size(b.U,3)), k max = $(nzp+NG)")
println("    Fx[i-NG+1,j-NG,k-NG,n]: i-NG+1 max = $(nxp+1), j-NG max = $(nyp), k-NG max = $(nzp)")
println("    Fx dims: $(size(Fx))")

# Check Areai bounds
println("    Areai[i+1,j,k]: i+1 max = $(nxp+NG+1), Areai dim1 = $(size(b.Areai,1))")
println("    nxi[i+1,j,k]: same...")

# Actually launch it
try
    @gpu_launch threads=threads_ri blocks=nb_ri Conser_reconstruct_i(b.Q, b.U, b.ϕ, b.Areai, Fx, b.Areai, b.nxi, b.nyi, b.nzi, nxp, nyp, nzp, b.stencil_i, b.Δstencil_i, b.lin_phi_i)
    CUDA.synchronize()
    println("  Conser_reconstruct_i: OK!")
    println("  Fx min/max: ", extrema(Array(Fx)))
catch e
    println("  Conser_reconstruct_i: FAILED! ", e)
end

println("\nDone.")
