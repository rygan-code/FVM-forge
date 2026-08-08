using LinearAlgebra
using StaticArrays

if !isdefined(@__MODULE__, :metric_ct_multiblock_point)
    include(joinpath(@__DIR__, "..", "METRIC_CT_MULTIBLOCK", "mesh_quality.jl"))
end

const CT_WENO7_INTERFACE_DIAGNOSTIC_TAG_BASE = 31000

function ct_weno7_validate_multiblock_topology(interfaces)
    length(interfaces) == 2 || error(
        "WENO7 multiblock acceptance requires exactly two reciprocal interfaces",
    )
    fields = map(_ct_weno7_interface_fields, interfaces)
    blocks = sort!(unique!([entry[1] for entry in fields]))
    blocks == [0, 1] || error(
        "WENO7 multiblock acceptance requires contiguous block IDs 0 and 1",
    )
    endpoints = [(entry[1], entry[2]) for entry in fields]
    length(unique(endpoints)) == 2 || error(
        "WENO7 multiblock acceptance contains duplicate local endpoints",
    )
    transforms = Dict{Tuple{Int,Int},Any}()
    for interface in interfaces
        transform = _ct_weno7_interface_transform(interface)
        transform === nothing && continue
        fields_entry = _ct_weno7_interface_fields(interface)
        transforms[(fields_entry[1], fields_entry[2])] = transform
    end
    for interface in interfaces
        _ct_weno7_validate_interface(interface)
    end
    for entry in fields
        block, local_face, neighbor_block, neighbor_face, reverse_tan = entry
        block != neighbor_block || error(
            "WENO7 multiblock interface cannot connect a block to itself",
        )
        reciprocal = (
            neighbor_block, neighbor_face, block, local_face, reverse_tan,
        )
        reciprocal in fields || error(
            "WENO7 multiblock interface is not reciprocal: $entry",
        )
        transform = get(transforms, (block, local_face), nothing)
        if transform !== nothing
            peer_transform = get(
                transforms, (neighbor_block, neighbor_face), nothing,
            )
            peer_transform === nothing ||
                peer_transform.source_for_destination ==
                structured_inverse_face_transform(transform).source_for_destination ||
                error(
                    "WENO7 multiblock transforms are not reciprocal: $entry",
                )
        end
    end
    return nothing
end

@inline function metric_ct_weno7_multiblock_point(
    logical::SVector{3,T},
) where {T}
    xi, eta, zeta = logical
    return SVector{3,T}(metric_ct_multiblock_point(xi, eta, zeta))
end

@inline function metric_ct_weno7_multiblock_covariants(
    logical::SVector{3,T},
) where {T}
    xi, eta, zeta = logical
    return (
        SVector{3,T}(
            one(T), T(0.10)*sin(eta/T(2))*cos(xi)*sin(zeta), zero(T),
        ),
        SVector{3,T}(
            zero(T), one(T) + T(0.20)*cos(eta) +
            T(0.05)*cos(eta/T(2))*sin(xi)*sin(zeta), zero(T),
        ),
        SVector{3,T}(
            zero(T), T(0.10)*sin(eta/T(2))*sin(xi)*cos(zeta), one(T),
        ),
    )
end

@inline function metric_ct_weno7_multiblock_face_normal(logical, direction)
    covariant = metric_ct_weno7_multiblock_covariants(logical)
    oriented = direction == 1 ? cross(covariant[2], covariant[3]) :
               direction == 2 ? cross(covariant[3], covariant[1]) :
                                cross(covariant[1], covariant[2])
    return normalize(oriented)
end

function ct_weno7_observe_interface_lines(blocks, plan, diagnostic_tag_base)
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_abs = 0.0
    local_rel = 0.0
    local_nonfinite = Int32(0)
    local_count = Int32(0)
    for (index, exchange) in enumerate(plan.exchanges)
        exchange.nb_rank != world_rank || error(
            "WENO7 multiblock observer requires remote interface peers",
        )
        length(plan.exchanges) == 1 || error(
            "WENO7 first-delivery observer supports one exchange per rank",
        )
        u_len = exchange.u_e - exchange.u_s + 1
        v_len = exchange.v_e - exchange.v_s + 1
        nvalue = ct_interface_buffer_length(u_len, v_len)
        send_buffer = Vector{FT}(undef, nvalue)
        recv_buffer = similar(send_buffer)
        ct_pack_interface_sheet!(send_buffer, blocks[exchange.bid], exchange)
        tag = _ct_checked_tag(
            diagnostic_tag_base, index - 1, CT_MPI_TAG_MAX,
            "WENO7 interface observer",
        )
        MPI.Sendrecv!(
            send_buffer, recv_buffer, MPI.COMM_WORLD;
            dest=exchange.nb_rank, sendtag=tag,
            source=exchange.nb_rank, recvtag=tag,
        )
        residual = ct_interface_line_residual(
            send_buffer, recv_buffer, u_len, v_len,
            exchange.fid, exchange.nb_fid, exchange.reverse_tan;
            transform=exchange.transform === nothing ? nothing :
                structured_inverse_face_transform(exchange.transform),
        )
        local_abs = max(local_abs, residual.absolute)
        local_rel = max(local_rel, residual.relative)
        local_nonfinite = max(local_nonfinite, residual.nonfinite_flag)
        local_count += Int32(1)
    end
    global_count = MPI.Allreduce(local_count, MPI.SUM, MPI.COMM_WORLD)
    global_count > 0 || error("WENO7 interface observer found no exchanges")
    return (
        absolute=MPI.Allreduce(local_abs, MPI.MAX, MPI.COMM_WORLD),
        relative=MPI.Allreduce(local_rel, MPI.MAX, MPI.COMM_WORLD),
        nonfinite_flag=MPI.Allreduce(
            local_nonfinite, MPI.MAX, MPI.COMM_WORLD,
        ),
    )
end

function metric_ct_weno7_exact_multiblock_interface_edge(
    direction, first_index, second_index, resolution,
)
    direction in (1, 3) || throw(ArgumentError(
        "interface edge direction must be 1 or 3",
    ))
    h = 2pi/resolution
    base = direction == 1 ?
        SVector(h*(first_index-1), pi, h*(second_index-1)) :
        SVector(h*(first_index-1), pi, h*(second_index-1))
    integral = 0.0
    for quadrature_index in eachindex(METRIC_CT_WENO7_EXACT_NODES)
        offset = (METRIC_CT_WENO7_EXACT_NODES[quadrature_index] + 1)/2
        logical = Base.setindex(
            base, base[direction] + h*offset, direction,
        )
        point = metric_ct_weno7_multiblock_point(logical)
        tangent = h*metric_ct_weno7_multiblock_covariants(logical)[direction]
        integral += METRIC_CT_WENO7_EXACT_WEIGHTS[quadrature_index] *
                    dot(ct_weno7_manufactured_electric(point), tangent)/2
    end
    return integral
end

@inline function _ct_weno7_interface_layer(index_a, size_a, index_b, size_b)
    distance = min(index_a - 1, size_a - index_a,
                   index_b - 1, size_b - index_b)
    return distance < 4 ? distance + 1 : 0
end

function ct_weno7_interface_error_summary(ex_edge, ez_edge, resolution)
    l1_sum = 0.0
    l2_sum = 0.0
    linf = 0.0
    count = 0
    layers = zeros(Float64, 4)
    for (direction, edge) in ((1, ex_edge), (3, ez_edge))
        size_a, size_b = size(edge)
        for second_index in 1:size_b, first_index in 1:size_a
            exact = metric_ct_weno7_exact_multiblock_interface_edge(
                direction, first_index, second_index, resolution,
            )
            error_value = abs(edge[first_index, second_index] - exact)
            isfinite(error_value) || error(
                "nonfinite WENO7 multiblock interface error",
            )
            layer = _ct_weno7_interface_layer(
                first_index, size_a, second_index, size_b,
            )
            if layer == 0
                l1_sum += error_value
                l2_sum += error_value^2
                linf = max(linf, error_value)
                count += 1
            else
                layers[layer] = max(layers[layer], error_value)
            end
        end
    end
    count > 0 || error(
        "WENO7 interface core is empty after excluding four layers",
    )
    return (
        l1_sum=l1_sum, l2_sum=l2_sum, linf=linf, count=count,
        layers=Tuple(layers),
    )
end

function ct_weno7_write_multiblock_stats(
    path, resolution, pre_sync, post_sync, errors, face_divb, nonfinite_flag,
)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(CT_WENO7_MULTIBLOCK_COLUMNS, " "))
        values = (
            resolution, pre_sync.absolute, pre_sync.relative,
            post_sync.absolute, post_sync.relative,
            errors.l1, errors.l2, errors.linf, errors.layers...,
            face_divb, nonfinite_flag,
        )
        println(io, join(values, " "))
    end
    return path
end
