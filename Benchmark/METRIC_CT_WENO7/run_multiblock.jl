using MPI
using HDF5

include(joinpath(@__DIR__, "run.jl"))
include(joinpath(ROOT, "ct_sync.jl"))
include(joinpath(ROOT, "ghost_coords.jl"))
include(joinpath(@__DIR__, "interface_diagnostics.jl"))

function multiblock_face_caches(resolution, block_id)
    h_xz = FT(2pi)/FT(resolution)
    h_eta = FT(pi)/FT(resolution)
    eta_origin = FT(block_id)*FT(pi)
    dims = (
        (resolution+1, resolution+2NG, resolution+2NG, 4),
        (resolution+2NG, resolution+1, resolution+2NG, 4),
        (resolution+2NG, resolution+2NG, resolution+1, 4),
    )
    return ntuple(3) do normal_direction
        cache = zeros(FT, dims[normal_direction])
        for k in axes(cache,3), j in axes(cache,2), i in axes(cache,1)
            storage = (i, j, k)
            logical = SVector{3,FT}(ntuple(3) do direction
                if direction == normal_direction
                    coordinate = FT(storage[direction]-1)
                else
                    coordinate = FT(storage[direction]-NG)-FT(0.5)
                end
                if direction == 2
                    eta_origin + h_eta*coordinate
                else
                    h_xz*coordinate
                end
            end)
            point = metric_ct_weno7_multiblock_point(logical)
            normal = metric_ct_weno7_multiblock_face_normal(
                logical, normal_direction,
            )
            electric = ct_weno7_manufactured_electric(point)
            tangential = electric-dot(electric, normal)*normal
            cache[i,j,k,1] = tangential[1]
            cache[i,j,k,2] = tangential[2]
            cache[i,j,k,3] = tangential[3]
            cache[i,j,k,4] = FT(0.5)
        end
        cache
    end
end

function multiblock_device_edges(resolution, block_id, host_coordinates)
    caches = map(CuArray, multiblock_face_caches(resolution, block_id))
    coordinates = map(CuArray, host_coordinates)
    scratch = CUDA.zeros(
        FT, resolution+2NG+1, resolution+2NG+1, resolution+2NG+1,
    )
    fail_meta = CUDA.zeros(Int32, CT_WENO7_FAIL_META_LEN)
    fail_value = CUDA.zeros(FT, 1)
    edges = (
        CUDA.zeros(FT, resolution, resolution+1, resolution+1),
        CUDA.zeros(FT, resolution+1, resolution, resolution+1),
        CUDA.zeros(FT, resolution+1, resolution+1, resolution),
    )
    for direction in 1:3
        ct_weno7_build_edge_line!(
            edges[direction], scratch, Val(direction), caches...,
            coordinates..., fail_meta, fail_value,
            MPI.Comm_rank(MPI.COMM_WORLD), block_id, Int32(1),
            resolution, resolution, resolution, (true, false, true),
        )
    end
    blocks = ntuple(_ -> Int32(cld(resolution+1, nthreads[1])), 3)
    @gpu_launch threads=nthreads blocks=blocks ct_sync_periodic_edge_emf_kernel!(
        edges..., true, false, true,
        Int32(resolution), Int32(resolution), Int32(resolution))
    gpu_sync()
    meta = Array(fail_meta)
    meta[CT_WENO7_FAIL_CLAIMED] == 0 || error(
        "WENO7 multiblock kernel recorded nonfinite metadata=$meta " *
        "value=$(Array(fail_value)[1])",
    )
    return edges, fail_meta
end

function multiblock_sync_state(edges, resolution, rank, topology)
    face_dims = ntuple(_ -> resolution + 2NG + 1, 3)
    block = (
        Nx=resolution, Ny=resolution, Nz=resolution,
        Bx_face=CUDA.zeros(FT, face_dims),
        By_face=CUDA.zeros(FT, face_dims),
        Bz_face=CUDA.zeros(FT, face_dims),
        Ex_edge=edges[1], Ey_edge=edges[2], Ez_edge=edges[3],
    )
    local_entries = filter(entry -> entry[1] == rank, topology)
    length(local_entries) == 1 || error(
        "expected one WENO7 interface entry for block/rank $rank",
    )
    _, fid, neighbor, neighbor_face, reverse_tan = only(local_entries)
    exchange = CTInterfaceExchange(
        rank, fid, neighbor, neighbor_face, neighbor, reverse_tan,
        1, resolution, 1, resolution,
        1, resolution, 1, resolution, CT_INTERFACE_TAG_BASE,
    )
    nvalue = ct_interface_buffer_length(resolution, resolution)
    nhalo = ct_interface_halo_buffer_length(resolution, resolution)
    plan = CTSyncPlan(
        [exchange], [zeros(FT, nvalue)], [zeros(FT, nvalue)],
        [zeros(FT, nhalo)], [zeros(FT, nhalo)], [0], 1,
    )
    return Dict(rank => block), plan
end

function load_multiblock_topology(mesh_directory)
    path = joinpath(mesh_directory, "block_connectivity.h5")
    return h5open(path, "r") do file
        connectivity = read(file["connectivity"])
        reverse_tan = vec(read(file["reverse_tan"]))
        size(connectivity, 1) == length(reverse_tan) || error(
            "connectivity/reverse_tan row count mismatch",
        )
        [(
            Int(connectivity[row,1]), Int(connectivity[row,2]),
            Int(connectivity[row,3]), Int(connectivity[row,4]),
            Bool(reverse_tan[row]),
        ) for row in axes(connectivity, 1)]
    end
end

function load_multiblock_coordinates(
    mesh_directory, resolution, block_id, topology,
)
    connectivity_path = joinpath(mesh_directory, "block_connectivity.h5")
    face_bc_matrix, nx_b, ny_b, nz_b, nblocks = h5open(
        connectivity_path, "r",
    ) do file
        read(file["face_bc"]), vec(read(file["Nx_b"])),
        vec(read(file["Ny_b"])), vec(read(file["Nz_b"])),
        Int(read(file["Nblocks"]))
    end
    nblocks == 2 || error("WENO7 multiblock mesh must contain two blocks")
    all(==(resolution), (nx_b..., ny_b..., nz_b...)) || error(
        "WENO7 multiblock mesh dimensions do not match N=$resolution",
    )

    mesh_path = joinpath(mesh_directory, "mesh_b$(block_id).h5")
    coords, mesh_ng, nx, ny, nz = h5open(mesh_path, "r") do file
        read(file["coords"]), Int(read(file["NG"])), Int(read(file["Nx"])),
        Int(read(file["Ny"])), Int(read(file["Nz"]))
    end
    mesh_ng == NG || error("WENO7 multiblock mesh NG=$mesh_ng, expected $NG")
    (nx, ny, nz) == (resolution, resolution, resolution) || error(
        "WENO7 block $block_id dimensions do not match N=$resolution",
    )
    size(coords) == (3, resolution+1, resolution+1, resolution+1) || error(
        "invalid coordinate array shape for WENO7 block $block_id",
    )

    face_bc = Dict(
        (bid, fid) => Int(face_bc_matrix[bid+1, fid])
        for bid in 0:1 for fid in 1:6
    )
    connectivity = Dict(
        (entry[1], entry[2]) => (
            src_b=entry[3], src_f=entry[4], reverse_tan=entry[5],
        ) for entry in topology
    )
    global mesh_dir = mesh_directory
    return expand_coords_with_ghost(
        coords[1,:,:,:], coords[2,:,:,:], coords[3,:,:,:],
        resolution, resolution, resolution, NG,
        face_bc, block_id, connectivity,
    )
end

function run_multiblock_case(resolution, mesh_directory, stats_path)
    MPI.Comm_size(MPI.COMM_WORLD) == 2 || error(
        "WENO7 multiblock acceptance requires exactly two MPI ranks",
    )
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    CUDA.functional() || error("WENO7 multiblock acceptance requires CUDA")
    devices = collect(CUDA.devices())
    isempty(devices) && error("no CUDA devices visible")
    CUDA.device!(devices[mod(rank, length(devices))+1])
    topology = load_multiblock_topology(mesh_directory)
    ct_weno7_validate_multiblock_topology(topology)
    coordinates = load_multiblock_coordinates(
        mesh_directory, resolution, rank, topology,
    )

    edges, fail_meta = multiblock_device_edges(resolution, rank, coordinates)
    blocks, plan = multiblock_sync_state(edges, resolution, rank, topology)
    pre_sync = ct_weno7_observe_interface_lines(
        blocks, plan, CT_WENO7_INTERFACE_DIAGNOSTIC_TAG_BASE,
    )
    local_edge_j = rank == 0 ? resolution + 1 : 1
    local_error = ct_weno7_interface_error_summary(
        Array(@view(edges[1][:,local_edge_j,:])),
        Array(@view(edges[3][:,local_edge_j,:])), resolution,
    )

    ct_sync_interface_sheets!(blocks, plan; sync_edges=true)
    post_sync = ct_weno7_observe_interface_lines(
        blocks, plan, CT_WENO7_INTERFACE_DIAGNOSTIC_TAG_BASE + 16,
    )
    global_l1_sum = MPI.Allreduce(local_error.l1_sum, MPI.SUM, MPI.COMM_WORLD)
    global_l2_sum = MPI.Allreduce(local_error.l2_sum, MPI.SUM, MPI.COMM_WORLD)
    global_count = MPI.Allreduce(local_error.count, MPI.SUM, MPI.COMM_WORLD)
    global_linf = MPI.Allreduce(local_error.linf, MPI.MAX, MPI.COMM_WORLD)
    global_layers = ntuple(4) do layer
        MPI.Allreduce(local_error.layers[layer], MPI.MAX, MPI.COMM_WORLD)
    end
    errors = (
        l1=global_l1_sum/global_count,
        l2=sqrt(global_l2_sum/global_count),
        linf=global_linf, layers=global_layers,
    )
    face_divb = MPI.Allreduce(
        repeated_rk3_stokes(map(Array, edges), resolution),
        MPI.MAX, MPI.COMM_WORLD,
    )
    nonfinite_flag = max(
        pre_sync.nonfinite_flag, post_sync.nonfinite_flag,
        Array(fail_meta)[CT_WENO7_FAIL_CLAIMED],
    )
    nonfinite_flag = MPI.Allreduce(nonfinite_flag, MPI.MAX, MPI.COMM_WORLD)
    @printf(
        "rank=%d cuda_device=%s N=%d pre_abs=%.16e post_abs=%.16e core=(%.16e,%.16e,%.16e) divB=%.3e\n",
        rank, string(CUDA.device()), resolution,
        pre_sync.absolute, post_sync.absolute,
        errors.l1, errors.l2, errors.linf, face_divb,
    )
    if rank == 0
        ct_weno7_write_multiblock_stats(
            stats_path, resolution, pre_sync, post_sync, errors,
            face_divb, nonfinite_flag,
        )
    end
    MPI.Barrier(MPI.COMM_WORLD)
    return nothing
end

function _run_multiblock_main(args)
    length(args) == 3 || error(
        "usage: run_multiblock.jl <resolution> <mesh_dir> <stats_path>",
    )
    MPI.Init()
    try
        run_multiblock_case(parse(Int, args[1]), args[2], args[3])
    finally
        MPI.Finalize()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    _run_multiblock_main(ARGS)
end
