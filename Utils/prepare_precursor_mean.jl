# =============================================================================
# prepare_precursor_mean.jl — Extract time-averaged cross-section for Fringe U_target
#
# Reads PLT snapshots from Phase 1 (non-rotating pipe flow),
# computes streamwise (x) + time average, and saves a (Ny, Nz, Ncons)
# conservative-variable profile for each block.
#
# Usage:
#   julia Utils/prepare_precursor_mean.jl <PLT_DIR> <MESH_DIR> [STEP_SPEC]
#
# Examples:
#   julia Utils/prepare_precursor_mean.jl PLT MESH_DIFFROT_COARSE last:20
#   julia Utils/prepare_precursor_mean.jl PLT MESH_DIFFROT_COARSE 10000:20000
#
# Output:
#   PLT/precursor_mean.h5   (contains U_mean_b{bid} for each block)
# =============================================================================

using HDF5, Statistics, Printf

# ─── Physical constants (must match run_pipe_diffrot.jl) ───
const γ  = 1.4
const Rg = 287.0

# ─── Scan available steps ───
function get_available_steps(plt_dir)
    files = readdir(plt_dir)
    plt_files = filter(f -> occursin(r"^plt-\d+-b0\.h5$", f), files)
    if isempty(plt_files)
        error("No plt-*-b0.h5 files found in $plt_dir/")
    end
    steps = [parse(Int, match(r"plt-(\d+)-b0", f).captures[1]) for f in plt_files]
    return sort(steps)
end

function parse_step_spec(spec_str, plt_dir)
    all_steps = get_available_steps(plt_dir)
    if occursin("last:", spec_str)
        n = parse(Int, split(spec_str, ":")[2])
        n = min(n, length(all_steps))
        return all_steps[end-n+1:end]
    elseif occursin(":", spec_str)
        parts = split(spec_str, ":")
        s1 = parse(Int, parts[1])
        s2 = parse(Int, parts[2])
        return filter(s -> s1 <= s <= s2, all_steps)
    else
        step = parse(Int, spec_str)
        return [step]
    end
end

"""
    prim_to_cons(rho, u, v, w, p)

Convert primitive variables to conservative variables:
  U = [ρ, ρu, ρv, ρw, ρE]
where E = p/(ρ(γ-1)) + 0.5(u²+v²+w²)
"""
function prim_to_cons(rho, u, v, w, p)
    rhoE = p ./ (γ - 1.0) .+ 0.5 .* rho .* (u.^2 .+ v.^2 .+ w.^2)
    return rho, rho .* u, rho .* v, rho .* w, rhoE
end

function main()
    if length(ARGS) < 2
        println("Usage: julia Utils/prepare_precursor_mean.jl <PLT_DIR> <MESH_DIR> [STEP_SPEC]")
        println("  STEP_SPEC: 'last:20', '10000:20000', or '15000'")
        return
    end

    plt_dir = ARGS[1]
    mesh_dir = ARGS[2]
    step_spec = length(ARGS) >= 3 ? ARGS[3] : "last:10"

    steps = parse_step_spec(step_spec, plt_dir)
    println("=" ^ 60)
    println("  Precursor Mean Profile Extraction")
    println("=" ^ 60)
    println("  PLT dir:    $plt_dir")
    println("  Mesh dir:   $mesh_dir")
    println("  Steps:      $(steps[1]) → $(steps[end]) ($(length(steps)) snapshots)")
    println()

    # Determine number of blocks from mesh directory
    Nblocks = 0
    while isfile(joinpath(mesh_dir, "mesh_b$(Nblocks).h5"))
        Nblocks += 1
    end
    println("  Blocks:     $Nblocks")

    # Read mesh dimensions for each block
    block_dims = Vector{Tuple{Int,Int,Int}}(undef, Nblocks)
    for bid in 0:Nblocks-1
        mesh_file = joinpath(mesh_dir, "mesh_b$(bid).h5")
        Nx = h5read(mesh_file, "Nx")
        Ny = h5read(mesh_file, "Ny")
        Nz = h5read(mesh_file, "Nz")
        block_dims[bid+1] = (Nx, Ny, Nz)
        println("  Block $bid:   $Nx × $Ny × $Nz")
    end
    println()

    Ncons = 5  # ρ, ρu, ρv, ρw, ρE

    # Accumulate x-averaged and time-averaged conservative variables
    # Result shape per block: (Ny, Nz, Ncons)
    U_mean = [zeros(Float64, Ny, Nz, Ncons) for (Nx, Ny, Nz) in block_dims]
    n_samples = 0

    for (si, step) in enumerate(steps)
        @printf("  Processing step %d (%d/%d)...\r", step, si, length(steps))

        for bid in 0:Nblocks-1
            Nx, Ny, Nz = block_dims[bid+1]
            plt_file = joinpath(plt_dir, "plt-$(step)-b$(bid).h5")
            if !isfile(plt_file)
                @warn "Missing: $plt_file, skipping"
                continue
            end

            # Read primitive variables
            fid = h5open(plt_file, "r")
            rho = Float64.(read(fid["rho"]))  # (Nx, Ny, Nz)
            u   = Float64.(read(fid["u"]))
            v   = Float64.(read(fid["v"]))
            w   = Float64.(read(fid["w"]))
            p   = Float64.(read(fid["p"]))
            close(fid)

            # Convert to conservative variables
            U1, U2, U3, U4, U5 = prim_to_cons(rho, u, v, w, p)

            # Average over x (streamwise) direction → (Ny, Nz)
            # Then accumulate for time averaging
            U_mean[bid+1][:, :, 1] .+= dropdims(mean(U1, dims=1), dims=1)
            U_mean[bid+1][:, :, 2] .+= dropdims(mean(U2, dims=1), dims=1)
            U_mean[bid+1][:, :, 3] .+= dropdims(mean(U3, dims=1), dims=1)
            U_mean[bid+1][:, :, 4] .+= dropdims(mean(U4, dims=1), dims=1)
            U_mean[bid+1][:, :, 5] .+= dropdims(mean(U5, dims=1), dims=1)
        end
        n_samples += 1
    end

    # Finalize time average
    for bid in 0:Nblocks-1
        U_mean[bid+1] ./= n_samples
    end

    println()
    println("  Averaged over $n_samples snapshots")

    # Save to HDF5
    out_path = joinpath(plt_dir, "precursor_mean.h5")
    h5open(out_path, "w") do fid
        fid["Nblocks"] = Nblocks
        fid["Ncons"] = Ncons
        fid["n_samples"] = n_samples
        fid["steps_used"] = collect(steps)
        fid["mesh_dir"] = mesh_dir

        for bid in 0:Nblocks-1
            Nx, Ny, Nz = block_dims[bid+1]
            grp = create_group(fid, "b$(bid)")
            grp["Ny"] = Ny
            grp["Nz"] = Nz
            grp["U_mean"] = Float32.(U_mean[bid+1])  # (Ny, Nz, Ncons)

            # Also save primitive means for diagnostics
            rho_m = U_mean[bid+1][:, :, 1]
            u_m   = U_mean[bid+1][:, :, 2] ./ rho_m
            v_m   = U_mean[bid+1][:, :, 3] ./ rho_m
            w_m   = U_mean[bid+1][:, :, 4] ./ rho_m
            E_m   = U_mean[bid+1][:, :, 5] ./ rho_m
            p_m   = (γ - 1.0) .* rho_m .* (E_m .- 0.5 .* (u_m.^2 .+ v_m.^2 .+ w_m.^2))
            grp["rho_mean"] = Float32.(rho_m)
            grp["u_mean"]   = Float32.(u_m)
            grp["v_mean"]   = Float32.(v_m)
            grp["w_mean"]   = Float32.(w_m)
            grp["p_mean"]   = Float32.(p_m)
        end
    end

    println("  Output:     $out_path")
    println()
    println("  Next step: set diffrot_phase = 2 in run_pipe_diffrot.jl")
    println("=" ^ 60)
end

main()
