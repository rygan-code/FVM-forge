# Pipe Flow Artifact Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a 128×44×44 pipe mesh with full-field artifact diagnostics, run locally and on cluster in parallel to detect numerical artifacts (>0.5%) after 2000+ steps.

**Architecture:** Extend existing debug infrastructure with a new mesh generator and enhanced full-field diagnostics. The test runner will scan every cell in the domain for checkerboard oscillations, gradient jumps, and solution boundedness violations.

**Tech Stack:** Julia, CUDA, MPI, HDF5, existing butterfly mesh generator

---

## File Structure

### Files to Create
- `debug/gen_pipe_test_mesh.jl` — Generate 128×44×44 butterfly pipe mesh
- `debug/run_pipe_artifact_test.jl` — Test runner with full-field diagnostics
- `debug/fullfield_diagnostic.jl` — Full-field artifact detection module

### Files to Modify
- None (all new files, no changes to production code)

### Files to Reference
- `Utils/gen_butterfly_fvm.jl` — Butterfly mesh generator (reuse)
- `debug/run_pipe_debug.jl` — Existing debug runner (template)
- `diag_checkerboard.jl` — Existing checkerboard diagnostic (extend)
- `solver.jl` — Core solver (reference only)
- `debug/MESH_DEBUG/` — Existing debug mesh (reference)

---

### Task 1: Create Mesh Generator (128×44×44)

**Files:**
- Create: `debug/gen_pipe_test_mesh.jl`
- Reference: `Utils/gen_butterfly_fvm.jl`

- [ ] **Step 1: Create mesh generator script**

```julia
# debug/gen_pipe_test_mesh.jl
# Generate 128×44×44 butterfly pipe mesh for artifact detection testing
# Usage: julia debug/gen_pipe_test_mesh.jl
# Outputs: debug/MESH_PIPE_TEST/

include("../Utils/gen_butterfly_fvm.jl")

# Override grid parameters for smaller test mesh
const Nx_test = 128   # streamwise (shorter than production 688)
const Ny_test = 44    # cross-section η
const Nz_test = 44    # cross-section ζ
const Lx_test = 2.0   # shorter pipe length (production is 7.5)

println("="^60)
println("  Generating Pipe Test Mesh")
println("  Grid: $(Nx_test)×$(Ny_test)×$(Nz_test)")
println("  Lx: $(Lx_test)")
println("="^60)

# Build mesh using existing generator
build_mesh("debug/MESH_PIPE_TEST", Nx_test, Ny_test, Nz_test, Lx_test)

println("Mesh generation complete: debug/MESH_PIPE_TEST/")
```

- [ ] **Step 2: Update build_mesh function to accept Lx parameter**

The existing `build_mesh` function in `Utils/gen_butterfly_fvm.jl` uses a global `Lx`. We need to make it parametric.

```julia
# Add to Utils/gen_butterfly_fvm.jl (after the existing build_mesh function)
function build_mesh(dir::String, Nx::Int, Ny::Int, Nz::Int, Lx::Float64)
    # Temporarily override global Lx
    global Lx_orig = Lx
    build_mesh(dir, Nx, Ny, Nz)
    global Lx = 7.5  # restore default
end
```

- [ ] **Step 3: Run mesh generator**

Run: `julia debug/gen_pipe_test_mesh.jl`
Expected: Mesh files created in `debug/MESH_PIPE_TEST/`

- [ ] **Step 4: Verify mesh files exist**

Run: `ls debug/MESH_PIPE_TEST/`
Expected: Files include `mesh_b0.h5` through `mesh_b4.h5`, `block_connectivity.h5`, `interp_weights.h5`

- [ ] **Step 5: Commit**

```bash
git add debug/gen_pipe_test_mesh.jl
git commit -m "feat: add 128×44×44 pipe mesh generator for artifact testing"
```

---

### Task 2: Create Full-Field Diagnostic Module

**Files:**
- Create: `debug/fullfield_diagnostic.jl`
- Reference: `diag_checkerboard.jl`

- [ ] **Step 1: Create full-field diagnostic module**

```julia
# debug/fullfield_diagnostic.jl
# Full-field artifact detection for pipe flow verification
# Scans every cell in the domain for checkerboard oscillations, gradient jumps,
# and solution boundedness violations.

export fullfield_diagnostic!

"""
    fullfield_diagnostic!(blocks, connectivity, tt_val, world_rank)

Run full-field artifact detection at specified step.
Checks:
1. Full-field checkerboard energy (2Δx oscillations)
2. Gradient smoothness (non-physical jumps)
3. Solution boundedness (physical values)
4. Interface vs interior comparison
"""
function fullfield_diagnostic!(blocks, connectivity, tt_val, world_rank)
    # Only trigger at specified step
    diag_step = @isdefined(fullfield_diag_step) ? fullfield_diag_step : 100
    if tt_val != diag_step; return; end
    
    if world_rank == 0
        @printf("\n╔═══════════════════════════════════════════════════════╗\n")
        @printf("║  FULL-FIELD ARTIFACT DETECTION — Step %d, Rank %d    ║\n", tt_val, world_rank)
        @printf("╚═══════════════════════════════════════════════════════╝\n")
    end
    
    ng = NG
    
    # Initialize global statistics
    global_max_e2dx = 0.0
    global_max_e2dx_loc = ""
    global_e2dx_sum = 0.0
    global_e2dx_count = 0
    
    interface_e2dx_sum = 0.0
    interface_e2dx_count = 0
    interior_e2dx_sum = 0.0
    interior_e2dx_count = 0
    
    # Solution bounds
    global_min_rho = Inf
    global_max_rho = -Inf
    global_min_p = Inf
    global_max_p = -Inf
    global_max_u = 0.0
    global_max_v = 0.0
    global_max_w = 0.0
    
    # Gradient smoothness
    global_max_drho = 0.0
    global_max_dp = 0.0
    global_max_drho_loc = ""
    global_max_dp_loc = ""
    
    # Process each block
    for (bid, b) in blocks
        U_cpu = Array(b.U)
        Nx, Ny, Nz = b.Nx, b.Ny, b.Nz
        
        # ── Full-field checkerboard energy (2Δx) ──
        for k in (ng+1):(Nz+ng), j in (ng+1):(Ny+ng), i in (ng+1):(Nx+ng)
            # Check all 3 directions
            for (di, dj, dk, dir_name) in [(1,0,0,"x"), (0,1,0,"y"), (0,0,1,"z")]
                i2 = i + di
                j2 = j + dj
                k2 = k + dk
                i3 = i + 2*di
                j3 = j + 2*dj
                k3 = k + 2*dk
                
                if i3 <= size(U_cpu,1) && j3 <= size(U_cpu,2) && k3 <= size(U_cpu,3)
                    # Density (variable 1)
                    val_c = U_cpu[i, j, k, 1]
                    val_p1 = U_cpu[i2, j2, k2, 1]
                    val_p2 = U_cpu[i3, j3, k3, 1]
                    
                    if abs(val_c) > 1e-10
                        e2dx = abs(val_c - 2*val_p1 + val_p2) / abs(val_c)
                        
                        # Track global maximum
                        if e2dx > global_max_e2dx
                            global_max_e2dx = e2dx
                            global_max_e2dx_loc = "Block $bid, i=$i, j=$j, k=$k, dir=$dir_name"
                        end
                        
                        # Accumulate for averaging
                        global_e2dx_sum += e2dx
                        global_e2dx_count += 1
                        
                        # Classify as interface or interior
                        is_interface = (j == ng+1 || j == Ny+ng || k == ng+1 || k == Nz+ng)
                        if is_interface
                            interface_e2dx_sum += e2dx
                            interface_e2dx_count += 1
                        else
                            interior_e2dx_sum += e2dx
                            interior_e2dx_count += 1
                        end
                    end
                end
            end
        end
        
        # ── Solution boundedness ──
        for k in (ng+1):(Nz+ng), j in (ng+1):(Ny+ng), i in (ng+1):(Nx+ng)
            rho = U_cpu[i, j, k, 1]
            u = U_cpu[i, j, k, 2] / rho
            v = U_cpu[i, j, k, 3] / rho
            w = U_cpu[i, j, k, 4] / rho
            E = U_cpu[i, j, k, 5]
            p = (γ - 1) * (E - 0.5 * rho * (u^2 + v^2 + w^2))
            T = p / (rho * Rg)
            
            global_min_rho = min(global_min_rho, rho)
            global_max_rho = max(global_max_rho, rho)
            global_min_p = min(global_min_p, p)
            global_max_p = max(global_max_p, p)
            global_max_u = max(global_max_u, abs(u))
            global_max_v = max(global_max_v, abs(v))
            global_max_w = max(global_max_w, abs(w))
        end
        
        # ── Gradient smoothness ──
        for k in (ng+1):(Nz+ng), j in (ng+1):(Ny+ng), i in (ng+1):(Nx+ng)
            rho_c = U_cpu[i, j, k, 1]
            
            # Check x-direction
            if i+1 <= size(U_cpu,1)
                rho_p = U_cpu[i+1, j, k, 1]
                if abs(rho_c) > 1e-10
                    drho = abs(rho_p - rho_c) / abs(rho_c)
                    if drho > global_max_drho
                        global_max_drho = drho
                        global_max_drho_loc = "Block $bid, i=$i, j=$j, k=$k, dir=x"
                    end
                end
            end
            
            # Check y-direction
            if j+1 <= size(U_cpu,2)
                rho_p = U_cpu[i, j+1, k, 1]
                if abs(rho_c) > 1e-10
                    drho = abs(rho_p - rho_c) / abs(rho_c)
                    if drho > global_max_drho
                        global_max_drho = drho
                        global_max_drho_loc = "Block $bid, i=$i, j=$j, k=$k, dir=y"
                    end
                end
            end
            
            # Check z-direction
            if k+1 <= size(U_cpu,3)
                rho_p = U_cpu[i, j, k+1, 1]
                if abs(rho_c) > 1e-10
                    drho = abs(rho_p - rho_c) / abs(rho_c)
                    if drho > global_max_drho
                        global_max_drho = drho
                        global_max_drho_loc = "Block $bid, i=$i, j=$j, k=$k, dir=z"
                    end
                end
            end
            
            # Pressure gradient
            p_c = (γ - 1) * (U_cpu[i, j, k, 5] - 0.5 * rho_c * 
                   (U_cpu[i, j, k, 2]^2 + U_cpu[i, j, k, 3]^2 + U_cpu[i, j, k, 4]^2) / rho_c)
            
            if i+1 <= size(U_cpu,1)
                rho_p = U_cpu[i+1, j, k, 1]
                p_p = (γ - 1) * (U_cpu[i+1, j, k, 5] - 0.5 * rho_p * 
                       (U_cpu[i+1, j, k, 2]^2 + U_cpu[i+1, j, k, 3]^2 + U_cpu[i+1, j, k, 4]^2) / rho_p)
                if abs(p_c) > 1e-10
                    dp = abs(p_p - p_c) / abs(p_c)
                    if dp > global_max_dp
                        global_max_dp = dp
                        global_max_dp_loc = "Block $bid, i=$i, j=$j, k=$k, dir=x"
                    end
                end
            end
        end
    end
    
    # ── Report results ──
    if world_rank == 0
        println("\n── Full-Field Checkerboard Energy ──")
        println("  Max E_2Δx: $(round(global_max_e2dx * 100, digits=6))% at $global_max_e2dx_loc")
        
        if global_e2dx_count > 0
            avg_e2dx = global_e2dx_sum / global_e2dx_count
            println("  Field average: $(round(avg_e2dx * 100, digits=6))%")
        end
        
        if interface_e2dx_count > 0
            avg_interface = interface_e2dx_sum / interface_e2dx_count
            println("  Interface average: $(round(avg_interface * 100, digits=6))%")
        end
        
        if interior_e2dx_count > 0
            avg_interior = interior_e2dx_sum / interior_e2dx_count
            println("  Interior average: $(round(avg_interior * 100, digits=6))%")
            
            if avg_interior > 0
                amplification = avg_interface / avg_interior
                println("  Amplification factor: $(round(amplification, digits=1))× (interface vs interior)")
            end
        end
        
        println("\n── Solution Bounds ──")
        println("  rho: [$(round(global_min_rho, digits=6)), $(round(global_max_rho, digits=6))]")
        println("  u:   [$(round(-global_max_u, digits=6)), $(round(global_max_u, digits=6))]")
        println("  v:   [$(round(-global_max_v, digits=6)), $(round(global_max_v, digits=6))]")
        println("  w:   [$(round(-global_max_w, digits=6)), $(round(global_max_w, digits=6))]")
        println("  p:   [$(round(global_min_p, digits=6)), $(round(global_max_p, digits=6))]")
        
        println("\n── Gradient Smoothness ──")
        println("  Max |Δρ/ρ|: $(round(global_max_drho * 100, digits=6))% at $global_max_drho_loc")
        println("  Max |Δp/p|: $(round(global_max_dp * 100, digits=6))% at $global_max_dp_loc")
        
        # Check for failures
        threshold = 0.005  # 0.5%
        failed = false
        
        if global_max_e2dx > threshold
            println("\n✗ FAIL: Checkerboard energy exceeds 0.5%")
            failed = true
        end
        
        if global_max_drho > threshold
            println("\n✗ FAIL: Density gradient exceeds 0.5%")
            failed = true
        end
        
        if global_max_dp > threshold
            println("\n✗ FAIL: Pressure gradient exceeds 0.5%")
            failed = true
        end
        
        if !failed
            println("\n✓ PASS: All artifacts < 0.5%")
        end
        
        # Write results to file
        fname = "debug/fullfield_diagnostic_step$(tt_val).txt"
        open(fname, "w") do io
            println(io, "Full-Field Artifact Detection Report")
            println(io, "====================================")
            println(io, "Step: $tt_val")
            println(io, "")
            println(io, "Checkerboard Energy:")
            println(io, "  Max E_2Δx: $(round(global_max_e2dx * 100, digits=6))%")
            println(io, "  Location: $global_max_e2dx_loc")
            println(io, "")
            println(io, "Solution Bounds:")
            println(io, "  rho: [$(round(global_min_rho, digits=6)), $(round(global_max_rho, digits=6))]")
            println(io, "  p:   [$(round(global_min_p, digits=6)), $(round(global_max_p, digits=6))]")
            println(io, "")
            println(io, "Gradient Smoothness:")
            println(io, "  Max |Δρ/ρ|: $(round(global_max_drho * 100, digits=6))%")
            println(io, "  Max |Δp/p|: $(round(global_max_dp * 100, digits=6))%")
            println(io, "")
            if failed
                println(io, "Result: FAIL")
            else
                println(io, "Result: PASS")
            end
        end
        println("\n  Results written to: $fname")
    end
    
    return nothing
end
```

- [ ] **Step 2: Commit**

```bash
git add debug/fullfield_diagnostic.jl
git commit -m "feat: add full-field artifact detection module"
```

---

### Task 3: Create Test Runner

**Files:**
- Create: `debug/run_pipe_artifact_test.jl`
- Reference: `debug/run_pipe_debug.jl`

- [ ] **Step 1: Create test runner script**

```julia
# debug/run_pipe_artifact_test.jl
# Pipe flow artifact detection test runner with full-field diagnostics
# Usage: julia debug/run_pipe_artifact_test.jl [nsteps]
# Default: 100 steps

const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100

println("="^60)
println("  Pipe Flow Artifact Detection Test")
println("  Steps: $(PROFILE_STEPS)")
println("="^60)

# ─── Global star-stencil controls ───
const limit_star::Bool = true
const TSVD_REL_TOL::Float64 = 1e-4
const TIKHONOV_LAMBDA::Float64 = 1e-4

# ─── Debug/Runtime flags ───
const debug_nan::Bool = true
const debug_sync::Bool = false
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

const flow_forcing::Bool = true
const forcing_mode::Int64 = 3
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = false

# ─── Spectral Warmup ───
const spectral_warmup::Bool = true
const spectral_warmup_order::Int64 = 4
const weno_z::Bool = true
const FT = Float64

# Thermal and Fluid parameters
const γ::FT = 1.4
const Rg::FT = 287
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = FT(1.458e-6)
const T_s::FT = 110.4
const Pr::FT = 0.71

# PipeFlow parameters (low Re/Ma for stability)
const Re_target::FT = FT(1000.0)
const Ma_target::FT = FT(0.3)
const Ro_target::FT = zero(FT)
const Tw::FT = FT(300.0)
const Lx::FT = FT(2.0)  # Shorter pipe for testing
const hit_forcing_A::FT = zero(FT)

# ─── GPU Backend ───
using CUDA

# ─── Physics / Equation System ───
const equation_type = :compressible
const _project_root = @__DIR__() * "/.."
include(joinpath(_project_root, "physics.jl"))
include(joinpath(_project_root, "solver.jl"))

# ─── LES (off) ───
const LES_smag::Bool = false
const LES_wale::Bool = false

# ─── Mesh ───
const mesh_dir = "debug/MESH_PIPE_TEST"
const connectivity_file = joinpath(mesh_dir, "block_connectivity.h5")

const _conf_data = h5open(connectivity_file, "r") do file
    nblocks = read(file["Nblocks"])
    nx_b = read(file["Nx_b"])
    ny_b = read(file["Ny_b"])
    nz_b = read(file["Nz_b"])
    ng = h5read(joinpath(mesh_dir, "mesh_b0.h5"), "NG")
    (nblocks, Tuple(nx_b), Tuple(ny_b), Tuple(nz_b), Int(ng))
end

const Nblocks::Int64 = _conf_data[1]
const Nx_b::NTuple{Nblocks, Int64} = _conf_data[2]
const Ny_b::NTuple{Nblocks, Int64} = _conf_data[3]
const Nz_b::NTuple{Nblocks, Int64} = _conf_data[4]
const NG::Int64 = _conf_data[5]
const R0::FT = FT(0.5)
const Omega_x::FT = zero(FT)
const x_rot_start::FT = zero(FT)
const x_rot_end::FT = Lx

# ─── Single GPU Execution Partition ───
const auto_partition_enabled::Bool = false
const gpu_vram_gb::Float64 = 8.0
MPI.Init()
include(joinpath(_project_root, "auto_partition.jl"))

const Block_Nprocs_manual = [SVector(1,1,1) for _ in 1:Nblocks]
const (Block_Nprocs, Block_to_rank) = (Block_Nprocs_manual, zeros(Int, Nblocks))

const Iperiodic = (true, false, false)

# ─── Flow control ───
const test_case::String = "PipeFlow"
const mesh::String = joinpath(mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(mesh_dir, "metrics_b0.h5")

const adaptive_dt::Bool = true
const CFL::FT = FT(0.3)
const LTS::Bool = false
const dt::FT = FT(1e-4)
const Time::FT = 100.0
const maxStep::Int64 = PROFILE_STEPS

const implicit::Bool = false
const implicit_CFL::FT = FT(10.0)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1e-3)

const plt_xdmf::Bool = false
const plt_out::Bool = true
const step_plt::Int64 = 100
const chk_out::Bool = false
const step_chk::Int64 = 1000
const restart::String = "none"
const inflow_restart::String = "none"

const average::Bool = false
const avg_step::Int64 = 10
const avg_total::Int64 = 1000
const avg_density_weighted::Bool = false

const sample::Bool = false
const sample_step::Int64 = 100
const sample_index::SVector{3, Int64} = [-1, -1, -1]

const filtering::Bool = true
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 1
const intf_filter_interval::Int64 = 1
const intf_filter_sigma::FT = FT(0.20)
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(0.01)

# ─── Checkerboard Diagnostic ───
const checkerboard_diag::Bool = true
const checkerboard_diag_step::Int64 = PROFILE_STEPS

# ─── Full-Field Diagnostic ───
const fullfield_diag::Bool = true
const fullfield_diag_step::Int64 = PROFILE_STEPS

# Include full-field diagnostic module
include("fullfield_diagnostic.jl")

const viscous::Bool = true
const viscous_order::Int64 = 6
const gg_blend::FT = one(FT)

# ─── Reconstruction ───
const eigen_reconstruction::Bool = true
const splitMethodID::Int32 = 4  # Roe
const hybrid_ϕ1::FT = FT(0.5)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = FT(0.2)
const UP7::SVector{7, FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7, FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7, FT} = UP7 * Linear_ϕ + CD6 * (one(FT) - Linear_ϕ)
const ΔLinear::SVector{7, FT} = UP7 - CD6

const maxreg::Int64 = 256
const nthreads::Tuple{Int32, Int32, Int32} = (8, 4, 8)
const nthreads2::Tuple{Int32, Int32, Int32} = (16, 8, 8)

# ─── Run Simulation ───
println("\n>>> Starting simulation...")
println("  Mesh: $mesh_dir")
println("  Blocks: $Nblocks")
println("  Grid: $(Nx_b[1])×$(Ny_b[1])×$(Nz_b[1])")
println("  Steps: $PROFILE_STEPS")
println("  Re: $Re_target")
println("  Ma: $Ma_target")

# Run the solver
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

t0 = time_ns()
time_step(rank, comm, Block_Nprocs)
wall = (time_ns() - t0) / 1e9

if rank == 0
    println("\n>>> Simulation completed in $(round(wall, digits=3)) s")
    println(">>> Check debug/fullfield_diagnostic_step$(PROFILE_STEPS).txt for results")
end

MPI.Finalize()
```

- [ ] **Step 2: Commit**

```bash
git add debug/run_pipe_artifact_test.jl
git commit -m "feat: add pipe flow artifact detection test runner"
```

---

### Task 4: Local Validation (Short Run)

**Files:**
- Modify: `debug/run_pipe_artifact_test.jl` (adjust steps)

- [ ] **Step 1: Run locally with 100 steps**

Run: `julia debug/run_pipe_artifact_test.jl 100`
Expected: Simulation completes, full-field diagnostic report generated

- [ ] **Step 2: Check diagnostic output**

Run: `cat debug/fullfield_diagnostic_step100.txt`
Expected: Report shows checkerboard energy, solution bounds, gradient smoothness

- [ ] **Step 3: Verify no NaN/Inf**

Check simulation output for any NaN/Inf errors. If found, investigate and fix.

- [ ] **Step 4: Commit results**

```bash
git add debug/fullfield_diagnostic_step100.txt
git commit -m "test: local validation with 100 steps"
```

---

### Task 5: Cluster Deployment (2000+ Steps)

**Files:**
- Create: `sub_artifact_test.sh` (SLURM job script)
- Upload: mesh, code, job script to cancon

- [ ] **Step 1: Create SLURM job script**

```bash
#!/bin/bash
#SBATCH -J artifact_test
#SBATCH -n 5
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-socket=1
#SBATCH -p kshdnormal
#SBATCH -o slurm-out-artifact
#SBATCH -e slurm-err-artifact

module purge
module load compiler/dtk/25.04.2 compiler/devtoolset/7.3.1 compiler/cmake/3.25.0 mpi/hpcx/2.7.4-gcc-7.3.1

mpirun -np 5 julia debug/run_pipe_artifact_test.jl 2000
```

- [ ] **Step 2: Upload files to cluster**

```powershell
# Upload mesh
scp -r debug/MESH_PIPE_TEST/ cancon:debug/MESH_PIPE_TEST/

# Upload code
scp debug/run_pipe_artifact_test.jl cancon:debug/
scp debug/fullfield_diagnostic.jl cancon:debug/
scp sub_artifact_test.sh cancon:/

# Upload solver dependencies
scp solver.jl physics.jl cancon:/
scp -r Utils/ cancon:Utils/
```

- [ ] **Step 3: Submit job**

```powershell
ssh cancon "sbatch sub_artifact_test.sh"
```

- [ ] **Step 4: Monitor job**

```powershell
ssh cancon "squeue -u ac6narhq4l"
ssh cancon "tail -50 slurm-out-artifact"
```

- [ ] **Step 5: Download results**

```powershell
scp cancon:debug/fullfield_diagnostic_step2000.txt ./
scp cancon:slurm-out-artifact ./
```

- [ ] **Step 6: Commit results**

```bash
git add debug/fullfield_diagnostic_step2000.txt
git commit -m "test: cluster validation with 2000 steps"
```

---

### Task 6: Analysis and Reporting

**Files:**
- Create: `debug/artifact_analysis_report.md`

- [ ] **Step 1: Analyze results**

Compare local (100 steps) vs cluster (2000 steps) results:
- Checkerboard energy trends
- Solution boundedness
- Gradient smoothness
- Interface vs interior amplification

- [ ] **Step 2: Generate analysis report**

```markdown
# Pipe Flow Artifact Detection Analysis

## Summary
- Local test (100 steps): [PASS/FAIL]
- Cluster test (2000 steps): [PASS/FAIL]

## Checkerboard Energy
- Max E_2Δx: [value]
- Interface average: [value]
- Interior average: [value]
- Amplification factor: [value]

## Solution Bounds
- Density: [range]
- Pressure: [range]
- Velocity: [range]

## Gradient Smoothness
- Max |Δρ/ρ|: [value]
- Max |Δp/p|: [value]

## Conclusion
[Summary of findings and recommendations]
```

- [ ] **Step 3: Commit analysis**

```bash
git add debug/artifact_analysis_report.md
git commit -m "docs: add artifact detection analysis report"
```

---

### Task 7: Failure Handling (If Artifacts Detected)

**Note:** This task is only executed if artifacts > 0.5% are detected.

**Files:**
- Create: `debug/artifact_fix_proposal.md`

- [ ] **Step 1: Analyze artifact patterns**

Identify root cause:
- Location (interface vs interior)
- Direction (x, y, z)
- Variable (density, pressure, velocity)
- Timing (when artifacts appear)

- [ ] **Step 2: Propose fix plan**

```markdown
# Artifact Fix Proposal

## Problem Statement
[Description of detected artifacts]

## Root Cause Analysis
[Analysis of why artifacts occur]

## Proposed Fixes

### Option 1: [Fix name]
- Description: [What the fix does]
- Files modified: [List of files]
- Risk: [Low/Medium/High]
- Expected improvement: [Quantitative estimate]

### Option 2: [Fix name]
- Description: [What the fix does]
- Files modified: [List of files]
- Risk: [Low/Medium/High]
- Expected improvement: [Quantitative estimate]

## Recommendation
[Which option to try first and why]

## Testing Plan
[How to verify the fix works]
```

- [ ] **Step 3: Present to user for approval**

Wait for user approval before implementing any fixes.

- [ ] **Step 4: Implement approved fix (if approved)**

Follow the approved fix plan from the proposal.

- [ ] **Step 5: Re-test after fix**

Run the same test suite to verify artifacts are eliminated.

- [ ] **Step 6: Commit fix**

```bash
git add [modified files]
git commit -m "fix: [description of fix]"
```

---

## Success Criteria

- [ ] All E_2Δx < 0.5% at block interfaces
- [ ] Solution remains bounded (physical values)
- [ ] No NaN/Inf through 2000 steps
- [ ] Full-field artifacts < 0.5%

## Dependencies

- Existing butterfly mesh generator (`Utils/gen_butterfly_fvm.jl`)
- Existing debug infrastructure (`debug/run_pipe_debug.jl`)
- Checkerboard diagnostic framework (`diag_checkerboard.jl`)
- Remote cluster access (cancon.hpccube.com)
