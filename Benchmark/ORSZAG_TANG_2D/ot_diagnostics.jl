const OT_STATS_COLUMNS = (
    :step, :time, :dt, :E_kin, :E_th, :E_mag, :E_Q, :E_cons,
    :E_Q_minus_E_cons, :min_rho_raw, :min_ei_raw, :min_p_raw,
    :divB_cell_L2, :divB_face_L2,
)

ot_stats_header() = "# " * join(string.(OT_STATS_COLUMNS), " ")

function ot_raw_mhd_minima(Uh, gamma)
    size(Uh, 4) >= 8 || throw(ArgumentError("U must contain at least 8 components"))
    min_rho = Inf
    min_ei = Inf
    min_p = Inf
    for k in axes(Uh, 3), j in axes(Uh, 2), i in axes(Uh, 1)
        raw = mhd_raw_thermo_components(
            Uh[i,j,k,1], Uh[i,j,k,2], Uh[i,j,k,3], Uh[i,j,k,4],
            Uh[i,j,k,5], Uh[i,j,k,6], Uh[i,j,k,7], Uh[i,j,k,8], gamma,
        )
        min_rho = min(min_rho, raw[1])
        min_ei = min(min_ei, raw[4])
        min_p = min(min_p, raw[5])
    end
    return (rho=min_rho, ei=min_ei, p=min_p)
end

function _ot_require_shape(name, array, expected)
    size(array) == expected || throw(DimensionMismatch(
        "$name has shape $(size(array)); expected $expected",
    ))
end

@inline function _ot_total_face_flux(face_flux, background_flux, i, j, k)
    value = @inbounds face_flux[i,j,k]
    return background_flux === nothing ? value :
           value + @inbounds(background_flux[i,j,k])
end

# CT stores point primitives in Q, while U contains cell averages. Combining
# U with Q[B] is therefore not a valid finite-volume state in POINT6 mode.
# Recover the magnetic cell average from the authoritative face fluxes using
# the same LSQ2 definition as the FOFC admissibility check.
function ot_raw_ct_mhd_minima(
    Uh, face_fluxes, face_metrics, gamma;
    background_face_fluxes=(nothing,nothing,nothing),
)
    size(Uh, 4) >= 5 || throw(ArgumentError("CT U must contain 5 components"))
    nx, ny, nz = size(Uh, 1), size(Uh, 2), size(Uh, 3)
    expected_shapes = (
        (nx+1,ny,nz), (nx,ny+1,nz), (nx,ny,nz+1),
    )
    for direction in 1:3
        _ot_require_shape(
            "face_fluxes[$direction]", face_fluxes[direction],
            expected_shapes[direction],
        )
        length(face_metrics[direction]) == 4 || throw(ArgumentError(
            "face_metrics[$direction] must contain area and three normals",
        ))
        for component in 1:4
            _ot_require_shape(
                "face_metrics[$direction][$component]",
                face_metrics[direction][component], expected_shapes[direction],
            )
        end
        background = background_face_fluxes[direction]
        background === nothing || _ot_require_shape(
            "background_face_fluxes[$direction]", background,
            expected_shapes[direction],
        )
    end

    Bx_face, By_face, Bz_face = face_fluxes
    B0x_face, B0y_face, B0z_face = background_face_fluxes
    (Areai,nxi,nyi,nzi), (Areaj,nxj,nyj,nzj),
        (Areak,nxk,nyk,nzk) = face_metrics
    min_rho = Inf
    min_ei = Inf
    min_p = Inf
    for k in axes(Uh, 3), j in axes(Uh, 2), i in axes(Uh, 1)
        @inbounds begin
            area_i_lo = SVector(
                Areai[i,j,k]*nxi[i,j,k],
                Areai[i,j,k]*nyi[i,j,k],
                Areai[i,j,k]*nzi[i,j,k],
            )
            area_i_hi = SVector(
                Areai[i+1,j,k]*nxi[i+1,j,k],
                Areai[i+1,j,k]*nyi[i+1,j,k],
                Areai[i+1,j,k]*nzi[i+1,j,k],
            )
            area_j_lo = SVector(
                Areaj[i,j,k]*nxj[i,j,k],
                Areaj[i,j,k]*nyj[i,j,k],
                Areaj[i,j,k]*nzj[i,j,k],
            )
            area_j_hi = SVector(
                Areaj[i,j+1,k]*nxj[i,j+1,k],
                Areaj[i,j+1,k]*nyj[i,j+1,k],
                Areaj[i,j+1,k]*nzj[i,j+1,k],
            )
            area_k_lo = SVector(
                Areak[i,j,k]*nxk[i,j,k],
                Areak[i,j,k]*nyk[i,j,k],
                Areak[i,j,k]*nzk[i,j,k],
            )
            area_k_hi = SVector(
                Areak[i,j,k+1]*nxk[i,j,k+1],
                Areak[i,j,k+1]*nyk[i,j,k+1],
                Areak[i,j,k+1]*nzk[i,j,k+1],
            )
        end
        magnetic = ct_recover_cell_b(
            area_i_lo,area_i_hi,area_j_lo,area_j_hi,
            area_k_lo,area_k_hi,
            _ot_total_face_flux(Bx_face,B0x_face,i,j,k),
            _ot_total_face_flux(Bx_face,B0x_face,i+1,j,k),
            _ot_total_face_flux(By_face,B0y_face,i,j,k),
            _ot_total_face_flux(By_face,B0y_face,i,j+1,k),
            _ot_total_face_flux(Bz_face,B0z_face,i,j,k),
            _ot_total_face_flux(Bz_face,B0z_face,i,j,k+1),
        )
        raw = mhd_raw_thermo_components(
            Uh[i,j,k,1], Uh[i,j,k,2], Uh[i,j,k,3], Uh[i,j,k,4],
            Uh[i,j,k,5], magnetic[1], magnetic[2], magnetic[3], gamma,
        )
        min_rho = min(min_rho, raw[1])
        min_ei = min(min_ei, raw[4])
        min_p = min(min_p, raw[5])
    end
    return (rho=min_rho, ei=min_ei, p=min_p)
end

function ot_read_stats(path)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    header_index = findfirst(
        line -> startswith(strip(line), "# step "), lines,
    )
    isnothing(header_index) && error("OT stats header is missing")
    names = Symbol.(split(replace(
        strip(lines[header_index]), r"^#\s*" => "",
    )))
    Tuple(names) == OT_STATS_COLUMNS || error(
        "OT stats schema mismatch: got $(Tuple(names))",
    )

    data_lines = filter(
        line -> !startswith(strip(line), "#"),
        lines[(header_index + 1):end],
    )
    isempty(data_lines) && error("OT stats contains no data rows")
    data = Matrix{Float64}(undef, length(data_lines), length(OT_STATS_COLUMNS))
    for (row, line) in enumerate(data_lines)
        values = parse.(Float64, split(strip(line)))
        length(values) == length(OT_STATS_COLUMNS) || error(
            "OT stats row $row has $(length(values)) columns",
        )
        data[row, :] .= values
    end
    return data
end

function ot_verify_stats(
    path;
    min_final_time=1.0,
    max_divb_face=1e-12,
    max_energy_drift=5e-3,
    min_timestep_ratio=1e-6,
)
    data = ot_read_stats(path)
    size(data, 1) >= 2 || error(
        "OT acceptance failed: expected an initial row and at least one post-step row",
    )
    all(isfinite, data) || error("OT acceptance failed: non-finite stats value")
    steps = view(data, :, 1)
    all(isinteger, steps) || error(
        "OT acceptance failed: steps must be exact integers",
    )
    all(steps .== (eachindex(steps) .- 1)) || error(
        "OT acceptance failed: steps must be consecutive from zero",
    )
    times = view(data, :, 2)
    timesteps = view(data, :, 3)
    times[1] == 0.0 || error("OT acceptance failed: initial time is not zero")
    timesteps[1] == 0.0 || error("OT acceptance failed: initial timestep is not zero")
    all(diff(times) .> 0) || error(
        "OT acceptance failed: post-step times are not strictly increasing",
    )
    post_timesteps = view(timesteps, 2:length(timesteps))
    all(post_timesteps .> 0) || error(
        "OT acceptance failed: non-positive post-step timestep",
    )
    for row in 2:length(times)
        observed_dt = times[row] - times[row - 1]
        tolerance = 16 * eps(Float64) * max(
            abs(times[row]), abs(times[row - 1]), abs(timesteps[row]), 1.0,
        )
        abs(observed_dt - timesteps[row]) <= tolerance || error(
            "OT acceptance failed: time increment at step $(steps[row]) does not match dt",
        )
    end
    times[end] >= min_final_time || error(
        "OT acceptance failed: final time $(times[end]) < $min_final_time",
    )
    minimum(view(data, :, 10)) > 0 || error(
        "OT acceptance failed: non-positive raw density",
    )
    minimum(view(data, :, 11)) > 0 || error(
        "OT acceptance failed: non-positive raw internal energy",
    )
    minimum(view(data, :, 12)) > 0 || error(
        "OT acceptance failed: non-positive raw pressure",
    )
    measured_divb = maximum(abs.(view(data, :, 14)))
    measured_divb <= max_divb_face || error(
        "OT acceptance failed: divB_face=$measured_divb > $max_divb_face",
    )
    e_cons = view(data, :, 8)
    e_cons[1] != 0 || error("OT acceptance failed: zero initial energy")
    measured_drift = maximum(abs.(e_cons ./ e_cons[1] .- 1))
    measured_drift <= max_energy_drift || error(
        "OT acceptance failed: energy drift=$measured_drift > $max_energy_drift",
    )
    dt_ratio = minimum(post_timesteps) / maximum(post_timesteps)
    dt_ratio >= min_timestep_ratio || error(
        "OT acceptance failed: timestep ratio=$dt_ratio < $min_timestep_ratio",
    )
    return (
        final_time=times[end],
        min_rho=minimum(view(data, :, 10)),
        min_ei=minimum(view(data, :, 11)),
        min_p=minimum(view(data, :, 12)),
        max_divb_face=measured_divb,
        max_energy_drift=measured_drift,
        timestep_ratio=dt_ratio,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia ot_diagnostics.jl <stats.dat>")
    println(ot_verify_stats(ARGS[1]))
end
