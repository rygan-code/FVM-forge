# Same-location and conservative self-convergence for nested warped OT meshes.
#
# Primitive PLT fields are point states associated with cell centers.  A coarse
# cell center is a fine-grid vertex, not a fine cell center, so direct odd-cell
# sampling is an O(h) position error.  The point comparison below interpolates
# fine cell-center data to the common coarse cell center with the same midpoint
# stencil used by the structured CT code.

using HDF5
using LinearAlgebra
using Printf
using StaticArrays

const RESOLUTIONS = (32, 64, 128)
const FIELDS = ("rho", "u", "v", "p", "Bx", "By")
const CONSERVATIVE_FIELDS = (
    ("rho_avg", 1),
    ("mom_x_avg", 2),
    ("mom_y_avg", 3),
    ("energy_avg", 5),
)
const MIDPOINT6_WEIGHTS = (
    3.0 / 256.0,
    -25.0 / 256.0,
    150.0 / 256.0,
    150.0 / 256.0,
    -25.0 / 256.0,
    3.0 / 256.0,
)

@inline function point(coords, i, j, k)
    return SVector(
        Float64(coords[1,i,j,k]),
        Float64(coords[2,i,j,k]),
        Float64(coords[3,i,j,k]),
    )
end

@inline quad_area(a, b, c, d) =
    0.5 * (cross(b-a, c-a) + cross(c-a, d-a))
@inline quad_center(a, b, c, d) = 0.25 * (a+b+c+d)

@inline periodic_index(index, extent) = mod(index - 1, extent) + 1

function cell_volumes(coords)
    nx, ny, nz = size(coords,2)-1, size(coords,3)-1, size(coords,4)-1
    volumes = Array{Float64}(undef, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        p000 = point(coords,i,j,k)
        p100 = point(coords,i+1,j,k)
        p010 = point(coords,i,j+1,k)
        p110 = point(coords,i+1,j+1,k)
        p001 = point(coords,i,j,k+1)
        p101 = point(coords,i+1,j,k+1)
        p011 = point(coords,i,j+1,k+1)
        p111 = point(coords,i+1,j+1,k+1)
        si_lo = quad_area(p000,p010,p011,p001)
        si_hi = quad_area(p100,p110,p111,p101)
        sj_lo = quad_area(p000,p001,p101,p100)
        sj_hi = quad_area(p010,p011,p111,p110)
        sk_lo = quad_area(p000,p100,p110,p010)
        sk_hi = quad_area(p001,p101,p111,p011)
        ci_lo = quad_center(p000,p010,p011,p001)
        ci_hi = quad_center(p100,p110,p111,p101)
        cj_lo = quad_center(p000,p001,p101,p100)
        cj_hi = quad_center(p010,p011,p111,p110)
        ck_lo = quad_center(p000,p100,p110,p010)
        ck_hi = quad_center(p001,p101,p111,p011)
        volumes[i,j,k] = (
            dot(ci_hi,si_hi)-dot(ci_lo,si_lo) +
            dot(cj_hi,sj_hi)-dot(cj_lo,sj_lo) +
            dot(ck_hi,sk_hi)-dot(ck_lo,sk_lo)
        ) / 3.0
    end
    minimum(volumes) > 0.0 || error("mesh contains non-positive cells")
    return volumes
end

function face_areas(coords)
    nx, ny, nz = size(coords,2)-1, size(coords,3)-1, size(coords,4)-1
    area_i = Array{Float64}(undef, nx+1, ny, nz)
    area_j = Array{Float64}(undef, nx, ny+1, nz)
    area_k = Array{Float64}(undef, nx, ny, nz+1)
    for k in 1:nz, j in 1:ny, i in 1:(nx+1)
        area_i[i,j,k] = norm(quad_area(
            point(coords,i,j,k), point(coords,i,j+1,k),
            point(coords,i,j+1,k+1), point(coords,i,j,k+1),
        ))
    end
    for k in 1:nz, j in 1:(ny+1), i in 1:nx
        area_j[i,j,k] = norm(quad_area(
            point(coords,i,j,k), point(coords,i,j,k+1),
            point(coords,i+1,j,k+1), point(coords,i+1,j,k),
        ))
    end
    for k in 1:(nz+1), j in 1:ny, i in 1:nx
        area_k[i,j,k] = norm(quad_area(
            point(coords,i,j,k), point(coords,i+1,j,k),
            point(coords,i+1,j+1,k), point(coords,i,j+1,k),
        ))
    end
    return (i=area_i, j=area_j, k=area_k)
end

function concatenate_cells(block0, block1)
    return cat(block0, block1; dims=2)
end

function concatenate_y_faces(block0, block1)
    # The two blocks share one y-face.  Average the duplicate only for the
    # diagnostic; a correct CT exchange makes the two values identical.
    interface = 0.5 .* (block0[:,end:end,:] .+ block1[:,1:1,:])
    return cat(block0[:,1:end-1,:], interface, block1[:,2:end,:]; dims=2)
end

function interpolate_nested_xy(fine)
    nx_f, ny_f, nz = size(fine)
    iseven(nx_f) && iseven(ny_f) ||
        error("fine grid is not 2:1 nested")
    nx, ny = nx_f ÷ 2, ny_f ÷ 2
    coarse = Array{Float64}(undef, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        i_left = 2*i - 1
        j_left = 2*j - 1
        value = 0.0
        for a in 1:6, b in 1:6
            ii = periodic_index(i_left + a - 3, nx_f)
            jj = periodic_index(j_left + b - 3, ny_f)
            value += MIDPOINT6_WEIGHTS[a] * MIDPOINT6_WEIGHTS[b] *
                     fine[ii,jj,k]
        end
        coarse[i,j,k] = value
    end
    return coarse
end

# Compatibility name for callers of the old analyzer.  It now means
# same-location interpolation, never direct odd-cell extraction.
sample_nested_xy(fine) = interpolate_nested_xy(fine)

function restrict_nested_cells(fine, fine_volumes)
    nx_f, ny_f, nz, ncomp = size(fine)
    iseven(nx_f) && iseven(ny_f) ||
        error("fine grid is not 2:1 nested")
    coarse = Array{Float64}(undef, nx_f ÷ 2, ny_f ÷ 2, nz, ncomp)
    for k in 1:nz, j in 1:(ny_f ÷ 2), i in 1:(nx_f ÷ 2)
        volume_sum = 0.0
        value = zeros(Float64, ncomp)
        for dj in 0:1, di in 0:1
            ii, jj = 2*i - 1 + di, 2*j - 1 + dj
            volume = fine_volumes[ii,jj,k]
            volume_sum += volume
            value .+= volume .* vec(fine[ii,jj,k,:])
        end
        coarse[i,j,k,:] .= value ./ volume_sum
    end
    return coarse
end

function restrict_i_face_flux(fine)
    nx_faces, ny, nz = size(fine)
    nx = nx_faces - 1
    iseven(nx) && iseven(ny) || error("fine face grid is not 2:1 nested")
    coarse = Array{Float64}(undef, nx ÷ 2 + 1, ny ÷ 2, nz)
    for k in 1:nz, j in 1:(ny ÷ 2), i in 1:(nx ÷ 2 + 1)
        ii = 2*i - 1
        coarse[i,j,k] = fine[ii,2*j-1,k] + fine[ii,2*j,k]
    end
    return coarse
end

function restrict_j_face_flux(fine)
    nx, ny_faces, nz = size(fine)
    ny = ny_faces - 1
    iseven(nx) && iseven(ny) || error("fine face grid is not 2:1 nested")
    coarse = Array{Float64}(undef, nx ÷ 2, ny ÷ 2 + 1, nz)
    for k in 1:nz, j in 1:(ny ÷ 2 + 1), i in 1:(nx ÷ 2)
        jj = 2*j - 1
        coarse[i,j,k] = fine[2*i-1,jj,k] + fine[2*i,jj,k]
    end
    return coarse
end

function read_checkpoint(path)
    h5open(path, "r") do file
        required = ("U", "Bx_face", "By_face", "Bz_face")
        all(haskey(file, name) for name in required) ||
            error("incomplete CT checkpoint: $path")
        names = (required..., "B0x_face", "B0y_face", "B0z_face")
        return Dict{String,Any}(
            name => Float64.(read(file[name]))
            for name in names if haskey(file, name)
        )
    end
end

function _total_face(state, name)
    background_name = replace(name, "Bx_face" => "B0x_face",
                              "By_face" => "B0y_face",
                              "Bz_face" => "B0z_face")
    return haskey(state, background_name) ?
        state[name] .+ state[background_name] : state[name]
end

function read_snapshot(case_dir, mesh_dir, resolution)
    plt_dir = joinpath(case_dir, "PLT")
    xmf_files = filter(name -> endswith(name, ".xmf"), readdir(plt_dir))
    isempty(xmf_files) && error("no snapshots found in $plt_dir")
    snapshot_steps = map(xmf_files) do name
        step_match = match(r"^plt-(\d+)\.xmf$",name)
        step_match === nothing && error("unexpected XMF name $name")
        parse(Int, step_match.captures[1])
    end
    xmf_name = xmf_files[argmax(snapshot_steps)]
    xmf_path = joinpath(plt_dir,xmf_name)
    time_match = match(r"<Time Value=\"([^\"]+)\"", read(xmf_path,String))
    time_match === nothing && error("snapshot time missing from $xmf_path")
    snapshot_time = parse(Float64,time_match.captures[1])
    step = maximum(snapshot_steps)
    stem = splitext(xmf_name)[1]
    block_data = Dict{Int,Dict{String,Array{Float64,3}}}()
    block_volumes = Dict{Int,Array{Float64,3}}()
    block_coords = Dict{Int,Array{Float64,4}}()
    block_states = Dict{Int,Union{Nothing,Dict{String,Any}}}()
    for bid in 0:1
        field_path = joinpath(plt_dir,"$(stem)-b$(bid).h5")
        mesh_path = joinpath(mesh_dir,"mesh_b$(bid).h5")
        fields = h5open(field_path,"r") do file
            Dict(field => Float64.(read(file[field])) for field in FIELDS)
        end
        expected_size = (resolution,div(resolution,2),8)
        all(size(values) == expected_size for values in values(fields)) ||
            error("unexpected field dimensions in $field_path")
        coords = h5open(mesh_path,"r") do file
            Float64.(read(file["coords"]))
        end
        checkpoint_path = joinpath(case_dir, "CHK", "chk-$step-b$bid.h5")
        state = isfile(checkpoint_path) ? read_checkpoint(checkpoint_path) : nothing
        block_data[bid] = fields
        block_volumes[bid] = cell_volumes(coords)
        block_coords[bid] = coords
        block_states[bid] = state
    end
    return (
        time=snapshot_time, step=step, fields=block_data,
        volumes=block_volumes, coords=block_coords, states=block_states,
    )
end

function pair_norm(coarse, fine, weights)
    difference = coarse .- fine
    if ndims(difference) == 3
        sum_volume = sum(weights)
        return (
            l1=sum(weights .* abs.(difference))/sum_volume,
            l2=sqrt(sum(weights .* abs2.(difference))/sum_volume),
            linf=maximum(abs,difference),
        )
    end
    weighted = reshape(weights, size(weights)..., 1)
    sum_volume = sum(weights)
    return (
        l1=sum(weighted .* abs.(difference))/sum_volume,
        l2=sqrt(sum(weighted .* abs2.(difference))/sum_volume),
        linf=maximum(abs,difference),
    )
end

function pair_error(coarse, fine, field)
    coarse_field = concatenate_cells(coarse.fields[0][field], coarse.fields[1][field])
    fine_field = concatenate_cells(fine.fields[0][field], fine.fields[1][field])
    restricted = interpolate_nested_xy(fine_field)
    coarse_volume = concatenate_cells(coarse.volumes[0], coarse.volumes[1])
    return pair_norm(coarse_field, restricted, coarse_volume)
end

function conservative_pair_error(coarse, fine, component)
    coarse_u = concatenate_cells(coarse.states[0]["U"], coarse.states[1]["U"])
    fine_u = concatenate_cells(fine.states[0]["U"], fine.states[1]["U"])
    fine_volume = concatenate_cells(fine.volumes[0], fine.volumes[1])
    coarse_volume = concatenate_cells(coarse.volumes[0], coarse.volumes[1])
    restricted = restrict_nested_cells(fine_u, fine_volume)
    return pair_norm(
        coarse_u[:,:,:,component], restricted[:,:,:,component], coarse_volume,
    )
end

function face_pair_error(coarse, fine, component)
    name = component === :i ? "Bx_face" : "By_face"
    coarse_flux = component === :i ?
        concatenate_cells(_total_face(coarse.states[0], name),
                          _total_face(coarse.states[1], name)) :
        concatenate_y_faces(_total_face(coarse.states[0], name),
                            _total_face(coarse.states[1], name))
    fine_flux = component === :i ?
        concatenate_cells(_total_face(fine.states[0], name),
                          _total_face(fine.states[1], name)) :
        concatenate_y_faces(_total_face(fine.states[0], name),
                            _total_face(fine.states[1], name))
    coarse_area_blocks = map((0,1)) do bid
        face_areas(coarse.coords[bid])[component]
    end
    coarse_area = component === :i ?
        concatenate_cells(coarse_area_blocks[1], coarse_area_blocks[2]) :
        concatenate_y_faces(coarse_area_blocks[1], coarse_area_blocks[2])
    restricted = component === :i ? restrict_i_face_flux(fine_flux) :
                                     restrict_j_face_flux(fine_flux)
    return pair_norm(coarse_flux ./ coarse_area,
                     restricted ./ coarse_area, coarse_area)
end

function worst_pair_difference(coarse, fine, field)
    coarse_field = concatenate_cells(coarse.fields[0][field], coarse.fields[1][field])
    fine_field = concatenate_cells(fine.fields[0][field], fine.fields[1][field])
    difference = abs.(coarse_field .- interpolate_nested_xy(fine_field))
    index = argmax(difference)
    i, j, k = Tuple(index)
    return (
        value=difference[index], index=(i,j,k),
        global_boundary_distance=(min(i-1,size(coarse_field,1)-i),
                                  min(j-1,size(coarse_field,2)-j)),
    )
end

function _orders(first, second)
    return (
        l1=log2(first.l1/second.l1),
        l2=log2(first.l2/second.l2),
        linf=log2(first.linf/second.linf),
    )
end

function main(run_root,case_suffix="",case_prefix="OT2D-CTHALO-WARP-2GPU")
    snapshots = Dict{Int,Any}()
    for resolution in RESOLUTIONS
        case_dir = joinpath(
            run_root,"results",
            "$(case_prefix)-N$(resolution)-T005$(case_suffix)",
        )
        mesh_dir = joinpath(
            run_root,"source","Benchmark","ORSZAG_TANG_2D_MPI_C1",
            "mesh","warped_N$(resolution)",
        )
        snapshots[resolution] = read_snapshot(case_dir,mesh_dir,resolution)
    end
    times = [snapshots[n].time for n in RESOLUTIONS]
    maximum(times)-minimum(times) <= 128eps(maximum(times)) || error(
        "snapshots are not at the same time: $times",
    )

    suffix_label = replace(lowercase(case_suffix),'-'=>'_')
    output_path = abspath(get(
        ENV, "OT2D_CONVERGENCE_OUTPUT",
        joinpath(run_root,"convergence_summary$(suffix_label).tsv"),
    ))
    open(output_path,"w") do io
        println(io,"field\tL1_32_64\tL1_64_128\tp_L1\tL2_32_64\tL2_64_128\tp_L2\tLinf_32_64\tLinf_64_128\tp_Linf")
        @printf("OT2D_CONVERGENCE_POINT time=%.16e\n",times[1])
        for field in FIELDS
            e1 = pair_error(snapshots[32],snapshots[64],field)
            e2 = pair_error(snapshots[64],snapshots[128],field)
            orders = _orders(e1,e2)
            @printf("%4s L1 %.6e -> %.6e p=%.3f  L2 %.6e -> %.6e p=%.3f  Linf %.6e -> %.6e p=%.3f\n",
                    field,e1.l1,e2.l1,orders.l1,e1.l2,e2.l2,orders.l2,
                    e1.linf,e2.linf,orders.linf)
            @printf(io,"%s\t%.16e\t%.16e\t%.8f\t%.16e\t%.16e\t%.8f\t%.16e\t%.16e\t%.8f\n",
                    field,e1.l1,e2.l1,orders.l1,e1.l2,e2.l2,orders.l2,
                    e1.linf,e2.linf,orders.linf)
            worst = worst_pair_difference(snapshots[64],snapshots[128],field)
            @printf("     worst64-128 ijk=%s boundary_distance=%s value=%.8e\n",
                    string(worst.index),string(worst.global_boundary_distance),worst.value)
        end
    end
    println("wrote $output_path")

    have_checkpoints = all(
        snapshots[n].states[0] !== nothing && snapshots[n].states[1] !== nothing
        for n in RESOLUTIONS
    )
    have_checkpoints || (@warn "CT checkpoints missing; conservative/face-flux convergence skipped")
    have_checkpoints || return nothing

    conservative_path = replace(output_path, ".tsv" => "_conservative.tsv")
    open(conservative_path,"w") do io
        println(io,"kind\tfield\tL1_32_64\tL1_64_128\tp_L1\tL2_32_64\tL2_64_128\tp_L2\tLinf_32_64\tLinf_64_128\tp_Linf")
        for (field, component) in CONSERVATIVE_FIELDS
            e1 = conservative_pair_error(snapshots[32],snapshots[64],component)
            e2 = conservative_pair_error(snapshots[64],snapshots[128],component)
            orders = _orders(e1,e2)
            println(io, join(("cell_average",field,e1.l1,e2.l1,orders.l1,
                              e1.l2,e2.l2,orders.l2,e1.linf,e2.linf,orders.linf), '\t'))
            @printf("cell_average %s L1 p=%.3f L2 p=%.3f Linf p=%.3f\n",
                    field,orders.l1,orders.l2,orders.linf)
        end
        for component in (:i, :j)
            e1 = face_pair_error(snapshots[32],snapshots[64],component)
            e2 = face_pair_error(snapshots[64],snapshots[128],component)
            orders = _orders(e1,e2)
            field = component === :i ? "Bn_i" : "Bn_j"
            println(io, join(("face_average",field,e1.l1,e2.l1,orders.l1,
                              e1.l2,e2.l2,orders.l2,e1.linf,e2.linf,orders.linf), '\t'))
            @printf("face_average %s L1 p=%.3f L2 p=%.3f Linf p=%.3f\n",
                    field,orders.l1,orders.l2,orders.linf)
        end
    end
    println("wrote $conservative_path")
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    1 <= length(ARGS) <= 3 || error(
        "usage: julia analyze_convergence.jl <run_root> [case_suffix] [case_prefix]",
    )
    main(abspath(ARGS[1]), length(ARGS) >= 2 ? ARGS[2] : "",
         length(ARGS) >= 3 ? ARGS[3] : "OT2D-CTHALO-WARP-2GPU")
end
