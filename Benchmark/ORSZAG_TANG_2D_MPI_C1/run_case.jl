# Case wrapper.  It reuses the existing Orszag-Tang solver entry without
# changing solver source files and redirects relative PLT output to one case
# directory by changing the working directory before include_string.

using Printf
using Serialization
using StaticArrays

const CASE_ID = get(ENV, "OT2D_CASE_ID", "OT2D-C1-UNNAMED")
const CASE_DIR = abspath(get(
    ENV, "OT2D_CASE_DIR", joinpath(@__DIR__, "cases", CASE_ID),
))
const MESH_DIR = abspath(get(
    ENV, "OT2D_MESH_DIR", joinpath(@__DIR__, "mesh", "warped_N32"),
))
const FINAL_TIME = get(ENV, "OT2D_FINAL_TIME", "0.1")
const CFL_VALUE = get(ENV, "OT_CFL", "0.2")
const CT_SCHEME = get(ENV, "OT2D_CT_SCHEME", "weno7")
const WENO7_EDGE_MODE = get(ENV, "OT2D_WENO7_EDGE_MODE", "hybrid")
const INITIAL_STATE_MODE = get(
    ENV, "OT2D_INITIAL_STATE_MODE", "quadrature6",
)
const SNAPSHOT_TIMES = get(ENV, "OT2D_SNAPSHOT_TIMES", "")
const SNAPSHOT_CHECKPOINTS = lowercase(get(
    ENV, "OT2D_SNAPSHOT_CHECKPOINTS", "true",
)) in ("1", "true", "yes", "on")
const FIXED_DT = get(ENV, "OT2D_FIXED_DT", "")
const HOST_NAME = get(ENV, "HOSTNAME", "unknown")
const STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
const MAX_STEPS_OVERRIDE = get(ENV, "OT2D_MAX_STEPS", "")
const INTERFACE_FILTER_ENABLED = lowercase(get(
    ENV, "OT2D_INTERFACE_FILTER", "true",
)) in ("1", "true", "yes", "on")
const interface_filter_enabled::Bool = INTERFACE_FILTER_ENABLED
const RIEMANN_ID = parse(Int32, get(ENV, "OT2D_RIEMANN_ID", "4"))
const DEBUG_INITIAL_FIELD = lowercase(get(
    ENV, "OT2D_DEBUG_INITIAL_FIELD", "false",
)) in ("1", "true", "yes", "on")
const DEBUG_INTERFACE_FLUX = lowercase(get(
    ENV, "OT2D_DEBUG_INTERFACE_FLUX", "false",
)) in ("1", "true", "yes", "on")
const DEBUG_STAGE_STEPS = Set(parse.(Int, filter(
    !isempty, split(get(ENV, "OT2D_DEBUG_STAGE_STEPS", "1"), ','),
)))
const DEBUG_STAGE_RKS = Set(parse.(Int, filter(
    !isempty, split(get(ENV, "OT2D_DEBUG_STAGE_RKS", "1"), ','),
)))
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ORIGINAL_RUNNER = joinpath(PROJECT_ROOT, "Benchmark", "ORSZAG_TANG_2D", "run.jl")

# The stock CT bootstrap averages cell-centered B onto a face.  That is not
# discretely divergence-free on a skew mesh.  Orszag-Tang has the periodic
# vector potential A_z = sqrt(mu0) * (cos(y) + cos(2x)/2), so initialize face
# fluxes through the solver's discrete-Stokes hook instead.
function in_situ_ct_initial_edge_integral_process(
    blocks, world_rank, Block_Nprocs, block_comms, metric_coordinates,
)
    vector_potential(x, y, z, time) = SVector{3,Float64}(
        0.0,
        0.0,
        SQRT_MU0_SI * (cos(y) + 0.5 * cos(2.0 * x)),
    )
    for block in values(blocks)
        ct_initial_edge_line_integrals_from_vector_potential!(
            block, vector_potential;
            coordinates=get(metric_coordinates, block.id, nothing),
            junction_fallback=(
                structured_metric_mode_setting() !=
                STRUCTURED_METRIC_LOCAL_CHART
            ),
        )
    end
    return nothing
end

function _ot2d_debug_initial_field(blocks, world_rank)
    DEBUG_INITIAL_FIELD || return nothing
    world_rank == 0 || return nothing
    for (bid, block) in blocks
        nx, ny, nz = block.Nx, block.Ny, block.Nz
        q_h = Array(block.Q)
        u_h = Array(block.U)
        vol_h = Array(block.Vol)
        x_h = Array(block.x)
        y_h = Array(block.y)
        max_node = 0.0
        max_center = 0.0
        e_q = 0.0
        e_u = 0.0
        first_sample = nothing
        for kk in (NG + 1):(NG + nz), jj in (NG + 1):(NG + ny), ii in (NG + 1):(NG + nx)
            x0 = Float64(x_h[ii, jj, kk])
            y0 = Float64(y_h[ii, jj, kk])
            dx = Float64(x_h[ii + 1, jj, kk] - x_h[ii, jj, kk])
            dy = Float64(y_h[ii, jj + 1, kk] - y_h[ii, jj, kk])
            bx = Float64(q_h[ii, jj, kk, 7])
            by = Float64(q_h[ii, jj, kk, 8])
            bx_node = -Float64(SQRT_MU0_SI) * sin(y0)
            by_node = Float64(SQRT_MU0_SI) * sin(2.0 * x0)
            bx_center = -Float64(SQRT_MU0_SI) * sin(y0 + 0.5 * dy)
            by_center = Float64(SQRT_MU0_SI) * sin(2.0 * (x0 + 0.5 * dx))
            max_node = max(max_node, abs(bx - bx_node), abs(by - by_node))
            max_center = max(max_center, abs(bx - bx_center), abs(by - by_center))
            rho = Float64(u_h[ii, jj, kk, 1])
            mx = Float64(u_h[ii, jj, kk, 2])
            my = Float64(u_h[ii, jj, kk, 3])
            mz = Float64(u_h[ii, jj, kk, 4])
            pressure = Float64(q_h[ii, jj, kk, 5])
            q_energy = 0.5 * (mx^2 + my^2 + mz^2) / rho +
                       pressure / (5.0 / 3.0 - 1.0) +
                       0.5 * Float64(INV_MU0_SI) * (bx^2 + by^2)
            cell_weight = 1.0 / Float64(vol_h[ii, jj, kk])
            e_q += q_energy * cell_weight
            e_u += Float64(u_h[ii, jj, kk, 5]) * cell_weight
            if first_sample === nothing
                first_sample = (x0=x0, y0=y0, bx=bx, by=by,
                                bx_node=bx_node, by_node=by_node,
                                bx_center=bx_center, by_center=by_center)
            end
        end
        println(
            "[OT2D-INIT-DEBUG] block=$bid max_abs_B_error_node=$(max_node) " *
            "max_abs_B_error_center=$(max_center) E_Q=$(e_q) E_U=$(e_u) " *
            "sample=$(first_sample)",
        )
    end
    flush(stdout)
    return nothing
end

function _ot2d_debug_interface_flux!(b, Fx, Fy, Fz, step, stage, world_rank)
    DEBUG_INTERFACE_FLUX || return nothing
    stage == 1 || return nothing
    step in (1, 2, 10, 100, 110) || return nothing
    gpu_sync()

    nx, ny, nz = b.Nx, b.Ny, b.Nz
    fy_h = Array(Fy)
    q_h = Array(b.Q)
    u_h = Array(b.U)
    i_range = 2:(nx + 1)
    k_range = 2:(nz + 1)
    fy_lo = sum(fy_h[ii, 1, kk, 5] for ii in i_range, kk in k_range)
    fy_hi = sum(fy_h[ii, ny + 1, kk, 5] for ii in i_range, kk in k_range)

    j_lo = NG + 1
    j_hi = ny + NG
    j_lo_ghost = NG
    j_hi_ghost = ny + NG + 1
    ii = NG + max(1, nx ÷ 2)
    kk = NG + max(1, nz ÷ 2)
    sample = (
        q_lo=(q_h[ii, j_lo, kk, 1], q_h[ii, j_lo, kk, 5],
              q_h[ii, j_lo, kk, 7], q_h[ii, j_lo, kk, 8],
              u_h[ii, j_lo, kk, 5]),
        q_lo_ghost=(q_h[ii, j_lo_ghost, kk, 1], q_h[ii, j_lo_ghost, kk, 5],
                    q_h[ii, j_lo_ghost, kk, 7], q_h[ii, j_lo_ghost, kk, 8],
                    u_h[ii, j_lo_ghost, kk, 5]),
        q_hi=(q_h[ii, j_hi, kk, 1], q_h[ii, j_hi, kk, 5],
              q_h[ii, j_hi, kk, 7], q_h[ii, j_hi, kk, 8],
              u_h[ii, j_hi, kk, 5]),
        q_hi_ghost=(q_h[ii, j_hi_ghost, kk, 1], q_h[ii, j_hi_ghost, kk, 5],
                    q_h[ii, j_hi_ghost, kk, 7], q_h[ii, j_hi_ghost, kk, 8],
                    u_h[ii, j_hi_ghost, kk, 5]),
    )
    local_interface_flux = b.id == 0 ? fy_hi : b.id == 1 ? fy_lo : 0.0
    interface_flux_sum = MPI.Allreduce(
        Float64(local_interface_flux), MPI.SUM, MPI.COMM_WORLD,
    )
    for rank in 0:(MPI.Comm_size(MPI.COMM_WORLD) - 1)
        if world_rank == rank
            println(
                "[OT2D-INTERFACE-FLUX] step=$step stage=$stage rank=$world_rank " *
                "block=$(b.id) fy_lo=$fy_lo fy_hi=$fy_hi " *
                "interface_sum=$interface_flux_sum sample=$sample",
            )
            flush(stdout)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
    return nothing
end

function _ot2d_debug_stage_snapshot!(
    b, label, step, stage, world_rank;
    Fx=nothing, Fy=nothing, Fz=nothing,
)
    DEBUG_INTERFACE_FLUX || return nothing
    step in DEBUG_STAGE_STEPS || return nothing
    stage in DEBUG_STAGE_RKS || return nothing
    gpu_sync()

    payload = Dict{Symbol,Any}(
        :label => String(label),
        :step => Int(step),
        :stage => Int(stage),
        :rank => Int(world_rank),
        :block => Int(b.id),
        :dimensions => (Int(b.Nx), Int(b.Ny), Int(b.Nz)),
        :U => Array(b.U),
        :Un => Array(b.Un),
        :Q => Array(b.Q),
        :Vol => Array(b.Vol),
        :Bx_face => Array(b.Bx_face),
        :By_face => Array(b.By_face),
        :Bz_face => Array(b.Bz_face),
        :Bx_face_n => Array(b.Bx_face_n),
        :By_face_n => Array(b.By_face_n),
        :Bz_face_n => Array(b.Bz_face_n),
        :Areai => Array(b.Areai),
        :nxi => Array(b.nxi),
        :nyi => Array(b.nyi),
        :nzi => Array(b.nzi),
        :Areaj => Array(b.Areaj),
        :nxj => Array(b.nxj),
        :nyj => Array(b.nyj),
        :nzj => Array(b.nzj),
        :Areak => Array(b.Areak),
        :nxk => Array(b.nxk),
        :nyk => Array(b.nyk),
        :nzk => Array(b.nzk),
        :Ex_edge => Array(b.Ex_edge),
        :Ey_edge => Array(b.Ey_edge),
        :Ez_edge => Array(b.Ez_edge),
    )
    b.fofc_flag === nothing || (payload[:fofc_flag] = Array(b.fofc_flag))
    Fx === nothing || (payload[:Fx] = Array(Fx))
    Fy === nothing || (payload[:Fy] = Array(Fy))
    Fz === nothing || (payload[:Fz] = Array(Fz))

    output_dir = joinpath(CASE_DIR, "stage_debug")
    mkpath(output_dir)
    filename = @sprintf(
        "step%06d_rk%d_%s_block%d_rank%d.jls",
        step, stage, label, b.id, world_rank,
    )
    open(joinpath(output_dir, filename), "w") do io
        serialize(io, payload)
    end
    return nothing
end

function _ot2d_debug_all_blocks!(blocks, label, step, stage, world_rank)
    DEBUG_INTERFACE_FLUX || return nothing
    for b in values(blocks)
        _ot2d_debug_stage_snapshot!(b, label, step, stage, world_rank)
    end
    return nothing
end

function _ot2d_debug_ghost_pair!(blocks, label, world_rank)
    DEBUG_INTERFACE_FLUX || return nothing
    world_rank == 0 || return nothing
    gpu_sync()
    haskey(blocks, 0) && haskey(blocks, 1) || return nothing
    b0 = blocks[0]
    b1 = blocks[1]
    u0 = Array(b0.U)
    u1 = Array(b1.U)
    q0 = Array(b0.Q)
    q1 = Array(b1.Q)
    i_range = (NG + 1):(NG + b0.Nx)
    k_range = (NG + 1):(NG + b0.Nz)
    j0_hi = NG + b0.Ny
    j1_lo = NG + 1
    j0_hi_ghost = j0_hi + 1
    j1_lo_ghost = NG
    for n in (1, 5)
        d_forward = maximum(abs.(
            u0[i_range, j0_hi_ghost, k_range, n] .-
            u1[i_range, j1_lo, k_range, n],
        ))
        d_reverse = maximum(abs.(
            u1[i_range, j1_lo_ghost, k_range, n] .-
            u0[i_range, j0_hi, k_range, n],
        ))
        println(
            "[OT2D-GHOST-DEBUG] label=$(label) U$(n) " *
            "b0_hi_ghost-vs-b1_lo_active=$(d_forward) " *
            "b1_lo_ghost-vs-b0_hi_active=$(d_reverse)",
        )
    end
    for n in (1, 5, 7, 8, 9)
        d_forward = maximum(abs.(
            q0[i_range, j0_hi_ghost, k_range, n] .-
            q1[i_range, j1_lo, k_range, n],
        ))
        d_reverse = maximum(abs.(
            q1[i_range, j1_lo_ghost, k_range, n] .-
            q0[i_range, j0_hi, k_range, n],
        ))
        println(
            "[OT2D-GHOST-DEBUG] label=$(label) Q$(n) " *
            "b0_hi_ghost-vs-b1_lo_active=$(d_forward) " *
            "b1_lo_ghost-vs-b0_hi_active=$(d_reverse)",
        )
    end
    flush(stdout)
    return nothing
end

isfile(ORIGINAL_RUNNER) || error("missing solver runner: $ORIGINAL_RUNNER")
isdir(MESH_DIR) || error("missing mesh directory: $MESH_DIR")
mkpath(CASE_DIR)

ENV["OT2D_MESH_DIR"] = MESH_DIR
ENV["OT2D_STATS_FILE"] = joinpath(CASE_DIR, "stats.dat")
ENV["OT2D_FINAL_TIME"] = FINAL_TIME
ENV["OT_FINAL_TIME"] = FINAL_TIME
ENV["OT_CFL"] = CFL_VALUE
ENV["OT2D_CT_SCHEME"] = CT_SCHEME
ENV["OT2D_INITIAL_STATE_MODE"] = INITIAL_STATE_MODE
ENV["OT2D_SNAPSHOT_CHECKPOINTS"] = string(SNAPSHOT_CHECKPOINTS)

open(joinpath(CASE_DIR, "manifest.txt"), "w") do io
    println(io, "case_id=$CASE_ID")
    println(io, "mesh_dir=$MESH_DIR")
    println(io, "case_dir=$CASE_DIR")
    println(io, "final_time=$FINAL_TIME")
    println(io, "cfl=$CFL_VALUE")
    println(io, "ct_scheme=$CT_SCHEME")
    println(io, "weno7_edge_mode=$WENO7_EDGE_MODE")
    println(io, "initial_state_mode=$INITIAL_STATE_MODE")
    println(io, "structured_scheduler=taskgraph")
    println(io, "snapshot_times=$SNAPSHOT_TIMES")
    println(io, "snapshot_checkpoints=$SNAPSHOT_CHECKPOINTS")
    println(io, "fixed_dt=$FIXED_DT")
    println(io, "ct_initial_face_flux=orszag_tang_vector_potential")
    println(io, "steps=$STEPS")
    println(io, "max_steps_override=$MAX_STEPS_OVERRIDE")
    println(io, "interface_filter_enabled=$INTERFACE_FILTER_ENABLED")
    println(io, "riemann_id=$RIEMANN_ID")
    println(io, "host=$HOST_NAME")
    println(io, "julia_version=$(VERSION)")
end

println("[OT2D-CASE] id=$CASE_ID backend=GPU mesh=$MESH_DIR case_dir=$CASE_DIR steps=$STEPS final_time=$FINAL_TIME")
flush(stdout)

empty!(ARGS)
push!(ARGS, string(STEPS))
cd(CASE_DIR)
source = read(ORIGINAL_RUNNER, String)

# Debug-only source controls.  These transformations keep the solver source
# untouched while allowing an A/B run to isolate the scheduler/filter and
# Riemann branches on exactly the same mesh and initialization.
const _OT2D_INTERFACE_FILTER_ENABLED = INTERFACE_FILTER_ENABLED
if !isempty(MAX_STEPS_OVERRIDE)
    max_steps = parse(Int, MAX_STEPS_OVERRIDE)
    max_steps >= 0 || error("OT2D_MAX_STEPS must be non-negative")
    source = replace(
        source,
        "const maxStep::Int64 = profiling ? PROFILE_STEPS : 100000" =>
        "const maxStep::Int64 = profiling ? PROFILE_STEPS : $max_steps",
    )
end
source = replace(
    source,
    "const splitMethodID::Int32 = 4" =>
    "const splitMethodID::Int32 = $RIEMANN_ID",
)
source = replace(
    source,
    "        0, activeTime, zero(FT), blocks,\n        world_rank, Block_Nprocs, block_comms,\n    )\nend\n\nfunction in_situ_post_process" =>
    "        0, activeTime, zero(FT), blocks,\n        world_rank, Block_Nprocs, block_comms,\n    )\n    _ot2d_debug_initial_field(blocks, world_rank)\nend\n\nfunction in_situ_post_process",
)

if DEBUG_INTERFACE_FLUX
    const _ot2d_source_lines = split(source, "\n")
    const _ot2d_include_hits = findall(
        line -> occursin("include(joinpath(_project_root", line) &&
                 occursin("structured_rk3_solver.jl", line),
        _ot2d_source_lines,
    )
    length(_ot2d_include_hits) == 1 || error(
        "OT2D debug hook: structured solver include was not found",
    )
    const _ot2d_solver_loader = join([
        "begin",
        "    _ot2d_solver_path = joinpath(_project_root, \"src\", \"time\", \"structured_rk3_solver.jl\")",
        "    _ot2d_solver_source = read(_ot2d_solver_path, String)",
        "    _ot2d_solver_lines = split(_ot2d_solver_source, \"\\n\")",
        "    for (_ot2d_callback, _ot2d_statement) in [",
        "        (\":u_interblock_face_copy\", \"            _ot2d_debug_ghost_pair!(blocks, \\\"after_first_U_copy\\\", world_rank)\"),",
        "        (\":u_interblock_full_copy\", \"            _ot2d_debug_ghost_pair!(blocks, \\\"after_full_U_copy\\\", world_rank); _ot2d_debug_all_blocks!(blocks, \\\"after_full_U_copy\\\", structured_task_sync_state[].step, structured_task_sync_state[].rk_stage, world_rank)\"),",
        "        (\":interface_filter\", \"            _ot2d_debug_all_blocks!(blocks, \\\"after_interface_filter\\\", structured_task_sync_state[].step, structured_task_sync_state[].rk_stage, world_rank)\"),",
        "        (\":filtered_u_interblock_full_copy\", \"            _ot2d_debug_all_blocks!(blocks, \\\"after_filtered_full_U_copy\\\", structured_task_sync_state[].step, structured_task_sync_state[].rk_stage, world_rank)\"),",
        "        (\":ct_point6_state\", \"            _ot2d_debug_all_blocks!(blocks, \\\"after_ct_point6_state\\\", structured_task_sync_state[].step, structured_task_sync_state[].rk_stage, world_rank)\"),",
        "    ]",
        "        _ot2d_header = findfirst(line -> occursin(\"structured_task_callbacks[\$(_ot2d_callback)]\", line), _ot2d_solver_lines)",
        "        _ot2d_header === nothing && error(\"OT2D debug hook: callback \$(_ot2d_callback) was not found\")",
        "        _ot2d_done = findfirst(idx -> idx > _ot2d_header && strip(_ot2d_solver_lines[idx]) == \"StructuredTaskDone\", eachindex(_ot2d_solver_lines))",
        "        _ot2d_done === nothing && error(\"OT2D debug hook: callback \$(_ot2d_callback) end was not found\")",
        "        insert!(_ot2d_solver_lines, _ot2d_done, _ot2d_statement)",
        "    end",
        "    _ot2d_task_flux = findfirst(line -> occursin(\"structured_rk_task_callbacks[:rk_flux]\", line), _ot2d_solver_lines)",
        "    _ot2d_task_flux === nothing && error(\"OT2D debug hook: task-graph flux callback was not found\")",
        "    _ot2d_task_flux_gate = findfirst(idx -> idx > _ot2d_task_flux && occursin(\"@static if strict_ct_positivity\", _ot2d_solver_lines[idx]), eachindex(_ot2d_solver_lines))",
        "    _ot2d_task_flux_gate === nothing && error(\"OT2D debug hook: task-graph flux callback end was not found\")",
        "    insert!(_ot2d_solver_lines, _ot2d_task_flux_gate, \"            _ot2d_debug_interface_flux!(b, shared_Fx, shared_Fy, shared_Fz, state[:tt], stage, world_rank)\")",
        "    insert!(_ot2d_solver_lines, _ot2d_task_flux_gate + 1, \"            _ot2d_debug_stage_snapshot!(b, \\\"post_flux\\\", state[:tt], stage, world_rank; Fx=shared_Fx, Fy=shared_Fy, Fz=shared_Fz)\")",
        "    for (_ot2d_callback, _ot2d_statement) in [",
        "        (\":rk_edge_emf\", \"            _ot2d_debug_stage_snapshot!(b, string(node.id, \\\"_post_edge_emf\\\"), state[:tt], stage, world_rank)\"),",
        "        (\":rk_fofc_detect\", \"            _ot2d_debug_stage_snapshot!(b, string(node.id, \\\"_post_detect\\\"), state[:tt], stage, world_rank; Fx=shared_Fx, Fy=shared_Fy, Fz=shared_Fz)\"),",
        "        (\":rk_divergence\", \"            _ot2d_debug_stage_snapshot!(b, string(node.id, \\\"_post_divergence\\\"), state[:tt], stage, world_rank)\"),",
        "        (\":rk_edge_sync\", \"            _ot2d_debug_all_blocks!(blocks, string(node.id, \\\"_post_edge_sync\\\"), structured_rk_task_state[][:tt], _structured_rk_stage_from_node(node), world_rank)\"),",
        "        (\":rk_junction_solve\", \"            _ot2d_debug_all_blocks!(blocks, string(node.id, \\\"_post_junction\\\"), structured_rk_task_state[][:tt], _structured_rk_stage_from_node(node), world_rank)\"),",
        "        (\":rk_face_b_update\", \"            _ot2d_debug_stage_snapshot!(b, string(node.id, \\\"_post_face_b_update\\\"), structured_rk_task_state[][:tt], stage, world_rank)\"),",
        "        (\":rk_face_barrier\", \"            _ot2d_debug_all_blocks!(blocks, string(node.id, \\\"_post_face_barrier\\\"), state[:tt], _structured_rk_stage_from_node(node), world_rank)\"),",
        "        (\":rk_stage_sync\", \"            _ot2d_debug_all_blocks!(blocks, string(node.id, \\\"_post_stage_sync\\\"), state[:tt], stage, world_rank)\"),",
        "    ]",
        "        _ot2d_header = findfirst(line -> occursin(\"structured_rk_task_callbacks[\$(_ot2d_callback)]\", line), _ot2d_solver_lines)",
        "        _ot2d_header === nothing && error(\"OT2D debug hook: callback \$(_ot2d_callback) was not found\")",
        "        _ot2d_done = findfirst(idx -> idx > _ot2d_header && strip(_ot2d_solver_lines[idx]) == \"StructuredTaskDone\", eachindex(_ot2d_solver_lines))",
        "        _ot2d_done === nothing && error(\"OT2D debug hook: callback \$(_ot2d_callback) end was not found\")",
        "        insert!(_ot2d_solver_lines, _ot2d_done, _ot2d_statement)",
        "    end",
        "    _ot2d_solver_source = join(_ot2d_solver_lines, \"\\n\")",
        "    include_string(Main, _ot2d_solver_source, _ot2d_solver_path)",
        "end",
    ], "\n")
    _ot2d_source_lines[_ot2d_include_hits[1]] = _ot2d_solver_loader
    source = join(_ot2d_source_lines, "\n")
end
include_string(Main, source, ORIGINAL_RUNNER)
