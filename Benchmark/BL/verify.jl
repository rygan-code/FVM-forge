using HDF5
using Printf

@isdefined(benchmark_plt_steps) ||
    include(joinpath(@__DIR__, "..", "benchmark_io.jl"))

function verify_bl(; plt_dir="PLT", mesh_path=joinpath(@__DIR__, "bl_mesh.h5"))
    latest = latest_benchmark_plt_file(plt_dir, 0)
    latest === nothing && return false
    _, output_path = latest
    isfile(mesh_path) || throw(ArgumentError("missing BL mesh: $mesh_path"))

    rho, u, p, temperature = h5open(output_path, "r") do file
        read(file["rho"]), read(file["u"]), read(file["p"]), read(file["T"])
    end
    coords = h5open(mesh_path, "r") do file
        read(file["coords"])
    end
    all(isfinite, rho) && all(isfinite, u) && all(isfinite, p) &&
        all(isfinite, temperature) || return false
    minimum(rho) > 0 && minimum(p) > 0 && minimum(temperature) > 0 ||
        return false

    nx, ny, _ = size(u)
    ny >= 2 || throw(DimensionMismatch("BL verification requires at least two y cells"))
    ix = round(Int, 0.75 * nx)
    1 <= ix <= nx || throw(BoundsError(1:nx, ix))
    x_position = coords[1, ix, 1, 1]
    u_profile = u[ix, :, 1]
    y_nodes = coords[2, ix, :, 1]
    y_cell = 0.5 .* (y_nodes[1:end-1] .+ y_nodes[2:end])

    u_inf = u_profile[end]
    yc1, yc2 = y_cell[1], y_cell[2]
    u1, u2 = u_profile[1], u_profile[2]
    dudy_wall = (u1 * yc2^2 - u2 * yc1^2) /
        (yc1 * yc2 * (yc2 - yc1))
    viscosity = 0.002366
    skin_friction = viscosity * dudy_wall /
        (0.5 * rho[ix, end, 1] * u_inf^2)
    reynolds_x = rho[ix, end, 1] * u_inf * x_position / viscosity
    theory = 0.664 / sqrt(reynolds_x)
    error_percent = abs(skin_friction - theory) / theory * 100

    @printf("BL: Cf=%.6e theory=%.6e error=%.2f%%\n",
            skin_friction, theory, error_percent)
    profile_path = joinpath(plt_dir, "bl_profile_data.csv")
    open(profile_path, "w") do io
        println(io, "y,u,u_norm")
        for j in 1:ny
            @printf(io, "%.6e,%.6e,%.6e\n",
                    y_cell[j], u_profile[j], u_profile[j] / u_inf)
        end
    end
    return isfinite(error_percent) && error_percent < 10.0
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    verify_bl() || exit(1)
end
