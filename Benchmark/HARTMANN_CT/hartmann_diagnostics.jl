using DelimitedFiles
using Printf

function hartmann_analytic_profiles(y, bulk_velocity, half_height, b0, mu, eta)
    hartmann = b0 * half_height / sqrt(mu * eta)
    denominator = 1 - tanh(hartmann) / hartmann
    velocity_scale = bulk_velocity / denominator
    velocity = velocity_scale .* (
        1 .- cosh.(hartmann .* y ./ half_height) ./ cosh(hartmann)
    )
    magnetic = (b0 * velocity_scale * half_height / (eta * hartmann)) .* (
        sinh.(hartmann .* y ./ half_height) ./ cosh(hartmann) .-
        (y ./ half_height) .* tanh(hartmann)
    )
    return velocity, magnetic, hartmann
end

function hartmann_profile(block)
    q = Array(block.Q)
    inverse_volume = Array(block.Vol)
    y_nodes = Array(block.y)
    y = zeros(Float64, block.Ny)
    velocity = zeros(Float64, block.Ny)
    magnetic = zeros(Float64, block.Ny)
    weights = zeros(Float64, block.Ny)
    for j in 1:block.Ny
        jj = NG + j
        y[j] = 0.5 * (y_nodes[NG + 1, jj, NG + 1] +
                      y_nodes[NG + 1, jj + 1, NG + 1])
        for k in 1:block.Nz, i in 1:block.Nx
            ii = NG + i
            kk = NG + k
            volume = 1 / inverse_volume[ii, jj, kk]
            velocity[j] += q[ii, jj, kk, 2] * volume
            magnetic[j] += q[ii, jj, kk, 7] * volume
            weights[j] += volume
        end
        velocity[j] /= weights[j]
        magnetic[j] /= weights[j]
    end
    bulk_velocity = sum(velocity .* weights) / sum(weights)
    return y, velocity, magnetic, weights, bulk_velocity
end

function hartmann_face_divergence(block, half_height, b0)
    bx_face = Array(block.Bx_face)
    by_face = Array(block.By_face)
    bz_face = Array(block.Bz_face)
    inverse_volume = Array(block.Vol)
    values = Float64[]
    weights = Float64[]
    for k in 1:block.Nz, j in 1:block.Ny, i in 1:block.Nx
        ii = NG + i
        jj = NG + j
        kk = NG + k
        divergence = inverse_volume[ii, jj, kk] * (
            bx_face[ii + 1, jj, kk] - bx_face[ii, jj, kk] +
            by_face[ii, jj + 1, kk] - by_face[ii, jj, kk] +
            bz_face[ii, jj, kk + 1] - bz_face[ii, jj, kk]
        )
        push!(values, divergence * half_height / b0)
        push!(weights, 1 / inverse_volume[ii, jj, kk])
    end
    l2 = sqrt(sum(weights .* abs2.(values)) / sum(weights))
    return l2, maximum(abs, values)
end

function hartmann_relative_errors(numerical, exact, weights; scale=maximum(abs, exact))
    error = numerical .- exact
    normalization = max(Float64(scale), eps(Float64))
    l1 = sum(weights .* abs.(error)) / sum(weights) / normalization
    l2 = sqrt(sum(weights .* abs2.(error)) / sum(weights)) / normalization
    linf = maximum(abs, error) / normalization
    return (l1=l1, l2=l2, linf=linf)
end

function verify_hartmann_ct(block; output_dir=@__DIR__)
    y, velocity, magnetic, weights, bulk_velocity = hartmann_profile(block)
    mu = get_viscosity(FT(Tw))
    exact_velocity, exact_magnetic, hartmann = hartmann_analytic_profiles(
        y, bulk_velocity, FT(R0), FT(B0), mu, FT(η_mhd),
    )
    velocity_errors = hartmann_relative_errors(
        velocity, exact_velocity, weights; scale=abs(bulk_velocity),
    )
    magnetic_errors = hartmann_relative_errors(magnetic, exact_magnetic, weights)
    divb_l2, divb_linf = hartmann_face_divergence(block, FT(R0), FT(B0))

    mkpath(output_dir)
    profile_path = joinpath(output_dir, "hartmann_ct_profile.dat")
    open(profile_path, "w") do io
        println(io, "# y u u_exact Bx Bx_exact")
        writedlm(io, hcat(y, velocity, exact_velocity, magnetic, exact_magnetic))
    end
    summary_path = joinpath(output_dir, "hartmann_ct_summary.dat")
    open(summary_path, "w") do io
        println(io, "# Ha bulk_u u_L1 u_L2 u_Linf bx_L1 bx_L2 bx_Linf divB_L2 divB_Linf")
        @printf(io, "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e\n",
            hartmann, bulk_velocity,
            velocity_errors.l1, velocity_errors.l2, velocity_errors.linf,
            magnetic_errors.l1, magnetic_errors.l2, magnetic_errors.linf,
            divb_l2, divb_linf)
    end

    @printf("HARTMANN_CT_RESULT Ha=%.8f bulk_u=%.8e ", hartmann, bulk_velocity)
    @printf("u_L1=%.8e u_L2=%.8e u_Linf=%.8e ",
        velocity_errors.l1, velocity_errors.l2, velocity_errors.linf)
    @printf("bx_L1=%.8e bx_L2=%.8e bx_Linf=%.8e ",
        magnetic_errors.l1, magnetic_errors.l2, magnetic_errors.linf)
    @printf("divB_L2=%.8e divB_Linf=%.8e\n", divb_l2, divb_linf)

    max_u_l2 = parse(Float64, get(ENV, "HARTMANN_CT_MAX_U_L2", "0.08"))
    max_b_l2 = parse(Float64, get(ENV, "HARTMANN_CT_MAX_B_L2", "0.15"))
    max_divb = parse(Float64, get(ENV, "HARTMANN_CT_MAX_DIVB", "1e-11"))
    velocity_errors.l2 <= max_u_l2 || error(
        "Hartmann velocity L2 error $(velocity_errors.l2) exceeds $max_u_l2",
    )
    magnetic_errors.l2 <= max_b_l2 || error(
        "Hartmann Bx L2 error $(magnetic_errors.l2) exceeds $max_b_l2",
    )
    divb_linf <= max_divb || error(
        "Hartmann face-divB Linf $divb_linf exceeds $max_divb",
    )
    return (
        hartmann=hartmann,
        bulk_velocity=bulk_velocity,
        velocity_errors=velocity_errors,
        magnetic_errors=magnetic_errors,
        divb_l2=divb_l2,
        divb_linf=divb_linf,
        profile_path=profile_path,
        summary_path=summary_path,
    )
end
