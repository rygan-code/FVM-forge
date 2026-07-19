using Printf

function resistive_ct_decay_metrics(block, time)
    q = Array(block.Q)
    inverse_volume = Array(block.Vol)
    x = Array(block.x)
    wave_number = 2pi / Float64(Lx)
    amplitude_sum = 0.0
    volume_sum = 0.0
    kinetic = 0.0
    internal = 0.0
    magnetic = 0.0
    divb_squared = 0.0
    divb_max = 0.0
    bx_face = Array(block.Bx_face)
    by_face = Array(block.By_face)
    bz_face = Array(block.Bz_face)

    for k in 1:block.Nz, j in 1:block.Ny, i in 1:block.Nx
        ii, jj, kk = NG+i, NG+j, NG+k
        volume = 1 / inverse_volume[ii,jj,kk]
        xc = 0.5 * (x[ii,jj,kk] + x[ii+1,jj,kk])
        sine, cosine = sincos(wave_number * xc)
        rho = q[ii,jj,kk,1]
        u, v, w = q[ii,jj,kk,2], q[ii,jj,kk,3], q[ii,jj,kk,4]
        pressure = q[ii,jj,kk,5]
        bx, by, bz = q[ii,jj,kk,7], q[ii,jj,kk,8], q[ii,jj,kk,9]
        amplitude_sum += volume * (by*sine + bz*cosine)
        volume_sum += volume
        kinetic += volume * 0.5*rho*(u*u + v*v + w*w)
        internal += volume * pressure / (Float64(γ) - 1)
        magnetic += volume * 0.5*(bx*bx + by*by + bz*bz)
        divb = inverse_volume[ii,jj,kk] * (
            bx_face[ii+1,jj,kk] - bx_face[ii,jj,kk] +
            by_face[ii,jj+1,kk] - by_face[ii,jj,kk] +
            bz_face[ii,jj,kk+1] - bz_face[ii,jj,kk]
        )
        divb_squared += volume * divb*divb
        divb_max = max(divb_max, abs(divb))
    end

    amplitude = amplitude_sum / volume_sum
    exact_amplitude = Float64(decay_amplitude) * exp(
        -Float64(η_mhd) * wave_number^2 * Float64(time),
    )
    initial_internal = volume_sum / (Float64(γ) - 1)
    initial_magnetic = 0.5 * Float64(decay_amplitude)^2 * volume_sum
    initial_total = initial_internal + initial_magnetic
    total = kinetic + internal + magnetic
    magnetic_loss = initial_magnetic - magnetic
    thermal_gain = internal - initial_internal
    closure_error = abs(thermal_gain + kinetic - magnetic_loss) /
                    max(abs(magnetic_loss), eps(Float64))
    return (
        amplitude=amplitude,
        exact_amplitude=exact_amplitude,
        amplitude_relative_error=abs(amplitude-exact_amplitude) /
                                 abs(exact_amplitude),
        kinetic=kinetic,
        internal=internal,
        magnetic=magnetic,
        total_relative_error=abs(total-initial_total) / initial_total,
        heating_closure_error=closure_error,
        divb_l2=sqrt(divb_squared / volume_sum),
        divb_linf=divb_max,
    )
end

function verify_resistive_ct_decay(block, time; output_dir=@__DIR__)
    metrics = resistive_ct_decay_metrics(block, time)
    mkpath(output_dir)
    path = joinpath(output_dir, "resistive_ct_decay_summary.dat")
    open(path, "w") do io
        println(io, "# time amplitude exact amplitude_rel kinetic internal magnetic total_rel heating_closure divB_L2 divB_Linf")
        @printf(io, "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e\n",
            time, metrics.amplitude, metrics.exact_amplitude,
            metrics.amplitude_relative_error, metrics.kinetic,
            metrics.internal, metrics.magnetic, metrics.total_relative_error,
            metrics.heating_closure_error, metrics.divb_l2, metrics.divb_linf)
    end
    @printf(
        "RESISTIVE_CT_DECAY_RESULT time=%.8e amplitude=%.8e exact=%.8e amplitude_rel=%.8e total_rel=%.8e heating_closure=%.8e divB_L2=%.8e divB_Linf=%.8e\n",
        time, metrics.amplitude, metrics.exact_amplitude,
        metrics.amplitude_relative_error, metrics.total_relative_error,
        metrics.heating_closure_error, metrics.divb_l2, metrics.divb_linf,
    )

    max_amplitude = parse(Float64, get(
        ENV, "RESISTIVE_CT_DECAY_MAX_AMPLITUDE_REL", "0.03",
    ))
    max_total = parse(Float64, get(
        ENV, "RESISTIVE_CT_DECAY_MAX_TOTAL_REL", "1e-10",
    ))
    max_closure = parse(Float64, get(
        ENV, "RESISTIVE_CT_DECAY_MAX_HEATING_CLOSURE", "0.03",
    ))
    max_divb = parse(Float64, get(
        ENV, "RESISTIVE_CT_DECAY_MAX_DIVB", "1e-11",
    ))
    metrics.amplitude_relative_error <= max_amplitude || error(
        "resistive CT amplitude error $(metrics.amplitude_relative_error) exceeds $max_amplitude",
    )
    metrics.total_relative_error <= max_total || error(
        "resistive CT total-energy error $(metrics.total_relative_error) exceeds $max_total",
    )
    metrics.heating_closure_error <= max_closure || error(
        "resistive CT heating closure $(metrics.heating_closure_error) exceeds $max_closure",
    )
    metrics.divb_linf <= max_divb || error(
        "resistive CT face-divB $(metrics.divb_linf) exceeds $max_divb",
    )
    return merge(metrics, (summary_path=path,))
end
