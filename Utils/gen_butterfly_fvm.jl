# =============================================================================
# Butterfly (O-H) Mesh Generator for Pipe Flow (FVM version)
# 5-block topology: 1 center square + 4 surrounding annular sectors
# Outputs REAL nodes only (no ghost cells). Ghost cells are generated at runtime.
# =============================================================================
using HDF5
using WriteVTK
using LinearAlgebra

if !isdefined(@__MODULE__, :StructuredFaceFrame)
    include(joinpath(@__DIR__, "..", "src", "core", "structured_interface_transform.jl"))
end

# ─── Global parameters ───
const NG::Int = 4
const Lx::Float64 = 7.5        # pipe length = 15R₀ (DNS standard)
const R0::Float64 = 0.5
const r_ratio::Float64 = 0.4
const r_inner::Float64 = r_ratio * R0
const vis::Bool = true
const compress_level::Int = 3
const FT = Float32  # Weight precision (matches solver runtime type)

# ─── Grid sizes ───
# DNS targets: Δx⁺<10, Δy⁺<0.5, rΔθ⁺<5
# u_τ ≈ 22.5 m/s, ν ≈ 0.01653 m²/s → ν/u_τ = 7.35e-4
# Δx = Lx/Nx < 10×7.35e-4 = 7.35e-3 → Nx > Lx/7.35e-3
# Δy_wall ≈ max_dr_outer < 0.5×7.35e-4 = 3.67e-4
# rΔθ = R₀(π/2)/Nz < 5×7.35e-4 = 3.67e-3 → Nz > R₀(π/2)/3.67e-3 ≈ 214
# Dimensions are now passed parametrically.

# =============================================================================
# Inner boundary mapping
# =============================================================================
function square_to_inner(s::Float64, t::Float64, r::Float64)
    alpha = 0.18
    ys = s * (1.0 + alpha * (1.0 - t^2))
    zt = t * (1.0 + alpha * (1.0 - s^2))
    return r * ys, r * zt
end

"""
    compute_tanh_param(ds_end, L_radial, N)

Solve for tanh stretching parameter `b` such that the end cell
has physical size `ds_end` across `N` cells spanning `L_radial`.
Uses Newton iteration on: 1 - tanh(b*(1-1/N))/tanh(b) = ds_end/L_radial
"""
function compute_tanh_param(ds_end::Float64, L_radial::Float64, N::Int)
    target = ds_end / L_radial   # normalized end-cell fraction
    ratio  = 1.0 - target         # = tanh(b*(1-1/N)) / tanh(b)
    s1     = 1.0 - 1.0 / N

    # Edge case: if target ≈ 1/N (uniform), return small b
    if abs(target - 1.0/N) < 1e-10
        return 0.01
    end

    b = 2.0  # initial guess
    for iter in 1:300
        tb  = tanh(b)
        tbs = tanh(b * s1)
        val = tbs / tb
        dval = (s1 / cosh(b * s1)^2 * tb - tbs / cosh(b)^2) / tb^2
        res  = val - ratio
        if abs(res) < 1e-14
            break
        end
        b -= res / dval
        b = clamp(b, 0.01, 100.0)
    end
    return b
end

"""
    two_sided_tanh(s, b_left, b_right)

Two-sided tanh stretching: clusters at both s=0 (b_left) and s=1 (b_right).
Uses a linear blend of two one-sided tanh functions.
f(0)=0, f(1)=1.  f'(0) controlled by b_left, f'(1) controlled by b_right.
"""
function two_sided_tanh(s::Float64, b_left::Float64, b_right::Float64)
    f_left  = 1.0 - tanh(b_left  * (1.0 - s)) / tanh(b_left)   # clusters at s=0
    f_right = tanh(b_right * s) / tanh(b_right)                   # clusters at s=1
    return (1.0 - s) * f_left + s * f_right
end

"""
    compute_corrected_b_right(b_left, ds_target, L_radial, N)

Iteratively find b_right so that the ACTUAL last cell of two_sided_tanh(s, b_left, b_right)
matches ds_target. This compensates for the blend distortion.
"""
function compute_corrected_b_right(b_left::Float64, ds_target::Float64, L_radial::Float64, N::Int)
    target_frac = ds_target / L_radial  # normalized target last-cell fraction
    # Initial guess from one-sided solution
    b_r = compute_tanh_param(ds_target, L_radial, N)
    
    for iter in 1:50
        # Compute actual last cell of blended function
        s_prev = Float64(N - 1) / N
        f_end  = 1.0  # f(1) = 1 always
        f_prev = two_sided_tanh(s_prev, b_left, b_r)
        actual_frac = f_end - f_prev
        
        # Adjust b_r: smaller b_r → larger last cell, larger b_r → smaller
        err = actual_frac - target_frac
        if abs(err) < 1e-12
            break
        end
        # Numerical derivative: d(actual_frac)/d(b_r)
        db = 1e-6
        f_prev_p = two_sided_tanh(s_prev, b_left, b_r + db)
        d_frac = ((1.0 - f_prev_p) - actual_frac) / db
        b_r -= err / d_frac
        b_r = clamp(b_r, 0.01, 100.0)
    end
    return b_r
end

# =============================================================================
# Grid Generation Functions — Real nodes only (Nx+1, Ny+1, Nz+1)
# =============================================================================

function generate_block0(Nx_b, Ny_b0, Nz_b0)
    Nx, Ny, Nz = Nx_b, Ny_b0, Nz_b0
    x = zeros(Float64, Nx+1, Ny+1, Nz+1)
    y = zeros(Float64, Nx+1, Ny+1, Nz+1)
    z = zeros(Float64, Nx+1, Ny+1, Nz+1)

    for k ∈ 1:Nz+1, j ∈ 1:Ny+1, i ∈ 1:Nx+1
        xa = Lx * Float64(i - 1) / Nx
        s_norm = 2.0 * Float64(j - 1) / Ny - 1.0
        t_norm = 2.0 * Float64(k - 1) / Nz - 1.0
        yc, zc = square_to_inner(s_norm, t_norm, r_inner)
        x[i, j, k] = xa
        y[i, j, k] = yc
        z[i, j, k] = zc
    end
    return x, y, z, Nx, Ny, Nz
end

function generate_block1(Nx_b, Ny_b0, Nz_b0, N_rad, dr_outer)
    Nx, Ny, Nz = Nx_b, N_rad, Nz_b0
    x = zeros(Float64, Nx+1, Ny+1, Nz+1)
    y = zeros(Float64, Nx+1, Ny+1, Nz+1)
    z = zeros(Float64, Nx+1, Ny+1, Nz+1)
    L_rad = R0 - r_inner
    # Match actual center block boundary cell (includes alpha=0.18 deformation)
    ds_inner = 2.0 * r_inner * (1.0 + 0.18) / Ny_b0
    b_wall  = compute_tanh_param(dr_outer, L_rad, Ny)  # wall at s=0
    b_inner = compute_corrected_b_right(b_wall, ds_inner, L_rad, Ny)  # inner at s=1 (corrected)
    
    for k ∈ 1:Nz+1, j ∈ 1:Ny+1, i ∈ 1:Nx+1
        xa = Lx * Float64(i - 1) / Nx
        s = Float64(j - 1) / Ny
        w_in = two_sided_tanh(s, b_wall, b_inner)  # wall s=0, inner s=1
        t_norm = 2.0 * Float64(k - 1) / Nz - 1.0
        y_in, z_in = square_to_inner(-1.0, t_norm, r_inner)
        param = Float64(k - 1) / Nz
        θ = 5*pi/4 + param * (pi/2)
        y_out = R0 * sin(θ); z_out = R0 * cos(θ)
        x[i, j, k] = xa
        y[i, j, k] = w_in * y_in + (1.0 - w_in) * y_out
        z[i, j, k] = w_in * z_in + (1.0 - w_in) * z_out
    end
    return x, y, z, Nx, Ny, Nz
end

function generate_block2(Nx_b, Ny_b0, Nz_b0, N_rad, dr_outer)
    Nx, Ny, Nz = Nx_b, N_rad, Nz_b0
    x = zeros(Float64, Nx+1, Ny+1, Nz+1)
    y = zeros(Float64, Nx+1, Ny+1, Nz+1)
    z = zeros(Float64, Nx+1, Ny+1, Nz+1)
    L_rad = R0 - r_inner
    ds_inner = 2.0 * r_inner * (1.0 + 0.18) / Ny_b0
    b_wall  = compute_tanh_param(dr_outer, L_rad, Ny)  # wall at s=1
    # For block2: inner at s=0, wall at s=1 → two_sided_tanh(s, b_inner, b_wall)
    # The first cell (inner) is the left end → correct as right via symmetry
    b_inner = compute_corrected_b_right(b_wall, ds_inner, L_rad, Ny)
    
    for k ∈ 1:Nz+1, j ∈ 1:Ny+1, i ∈ 1:Nx+1
        xa = Lx * Float64(i - 1) / Nx
        s = Float64(j - 1) / Ny
        w_in = 1.0 - two_sided_tanh(s, b_inner, b_wall)  # inner s=0, wall s=1
        t_norm = 2.0 * Float64(k - 1) / Nz - 1.0
        y_in, z_in = square_to_inner(1.0, t_norm, r_inner)
        param = Float64(k - 1) / Nz
        θ = 3*pi/4 - param * (pi/2)
        y_out = R0 * sin(θ); z_out = R0 * cos(θ)
        x[i, j, k] = xa
        y[i, j, k] = w_in * y_in + (1.0 - w_in) * y_out
        z[i, j, k] = w_in * z_in + (1.0 - w_in) * z_out
    end
    return x, y, z, Nx, Ny, Nz
end

function generate_block3(Nx_b, Ny_b0, Nz_b0, N_rad, dr_outer)
    Nx, Ny, Nz = Nx_b, Ny_b0, N_rad
    x = zeros(Float64, Nx+1, Ny+1, Nz+1)
    y = zeros(Float64, Nx+1, Ny+1, Nz+1)
    z = zeros(Float64, Nx+1, Ny+1, Nz+1)
    L_rad = R0 - r_inner
    ds_inner = 2.0 * r_inner * (1.0 + 0.18) / Nz_b0
    b_wall  = compute_tanh_param(dr_outer, L_rad, Nz)  # wall at s=0
    b_inner = compute_corrected_b_right(b_wall, ds_inner, L_rad, Nz)  # inner at s=1 (corrected)
    
    for k ∈ 1:Nz+1, j ∈ 1:Ny+1, i ∈ 1:Nx+1
        xa = Lx * Float64(i - 1) / Nx
        s = Float64(k - 1) / Nz
        w_in = two_sided_tanh(s, b_wall, b_inner)  # wall s=0, inner s=1
        s_norm = 2.0 * Float64(j - 1) / Ny - 1.0
        y_in, z_in = square_to_inner(s_norm, -1.0, r_inner)
        param = Float64(j - 1) / Ny
        θ = 5*pi/4 - param * (pi/2)
        y_out = R0 * sin(θ); z_out = R0 * cos(θ)
        x[i, j, k] = xa
        y[i, j, k] = w_in * y_in + (1.0 - w_in) * y_out
        z[i, j, k] = w_in * z_in + (1.0 - w_in) * z_out
    end
    return x, y, z, Nx, Ny, Nz
end

function generate_block4(Nx_b, Ny_b0, Nz_b0, N_rad, dr_outer)
    Nx, Ny, Nz = Nx_b, Ny_b0, N_rad
    x = zeros(Float64, Nx+1, Ny+1, Nz+1)
    y = zeros(Float64, Nx+1, Ny+1, Nz+1)
    z = zeros(Float64, Nx+1, Ny+1, Nz+1)
    L_rad = R0 - r_inner
    ds_inner = 2.0 * r_inner * (1.0 + 0.18) / Nz_b0
    b_wall  = compute_tanh_param(dr_outer, L_rad, Nz)  # wall at s=1
    b_inner = compute_corrected_b_right(b_wall, ds_inner, L_rad, Nz)
    
    for k ∈ 1:Nz+1, j ∈ 1:Ny+1, i ∈ 1:Nx+1
        xa = Lx * Float64(i - 1) / Nx
        s = Float64(k - 1) / Nz
        w_in = 1.0 - two_sided_tanh(s, b_inner, b_wall)  # inner s=0, wall s=1
        s_norm = 2.0 * Float64(j - 1) / Ny - 1.0
        y_in, z_in = square_to_inner(s_norm, 1.0, r_inner)
        param = Float64(j - 1) / Ny
        θ = 7*pi/4 + param * (pi/2)
        y_out = R0 * sin(θ); z_out = R0 * cos(θ)
        x[i, j, k] = xa
        y[i, j, k] = w_in * y_in + (1.0 - w_in) * y_out
        z[i, j, k] = w_in * z_in + (1.0 - w_in) * z_out
    end
    return x, y, z, Nx, Ny, Nz
end

# =============================================================================
# VTK Export (real nodes only)
# =============================================================================
function export_mesh_vtk(blocks_data, out_dir)
    for bid in 0:4
        x, y, z, Nx, Ny, Nz = blocks_data[bid + 1]
        vtk_grid(joinpath(out_dir, "mesh_debug_b$bid"), x, y, z) do vtk
        end
    end
    
    # Manually create VTM file
    vtm_path = joinpath(out_dir, "mesh_debug_all.vtm")
    open(vtm_path, "w") do f
        write(f, "<?xml version=\"1.0\"?>\n")
        write(f, "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n")
        write(f, "  <vtkMultiBlockDataSet>\n")
        for bid in 0:4
            write(f, "    <DataSet index=\"$bid\" name=\"Block_$bid\" file=\"mesh_debug_b$bid.vts\"/>\n")
        end
        write(f, "  </vtkMultiBlockDataSet>\n")
        write(f, "</VTKFile>\n")
    end
end


function build_mesh(out_dir, Nx_b, Ny_b0, Nz_b0, N_rad)
    println("Generating FVM Butterfly Mesh into ", out_dir, " ...")
    mkpath(out_dir)
    blocks_data = []
    
    max_dr_outer = 0.00035 * (1024.0 / Nx_b)
    
    push!(blocks_data, generate_block0(Nx_b, Ny_b0, Nz_b0))
    push!(blocks_data, generate_block1(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer))
    push!(blocks_data, generate_block2(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer))
    push!(blocks_data, generate_block3(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer))
    push!(blocks_data, generate_block4(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer))
    
    for bid in 0:4
        x, y, z, Nx, Ny, Nz = blocks_data[bid + 1]
        h5open(joinpath(out_dir, "mesh_b$bid.h5"), "w") do f
            f["NG"] = NG
            f["Nx"] = Int64(Nx)
            f["Ny"] = Int64(Ny)
            f["Nz"] = Int64(Nz)
            f["coords"] = Float32.(cat(reshape(x, (1, size(x)...)), reshape(y, (1, size(y)...)), reshape(z, (1, size(z)...)), dims=1))
            # Separate x/y/z datasets for XDMF X_Y_Z geometry (most compatible)
            f["x"] = Float32.(x)
            f["y"] = Float32.(y)
            f["z"] = Float32.(z)
        end
    end
    
    # ─── Connectivity with auto-computed reverse_tan ───
    connections = Dict(
        (0, 3) => (1, 4), (1, 4) => (0, 3),
        (0, 4) => (2, 3), (2, 3) => (0, 4),
        (0, 5) => (3, 6), (3, 6) => (0, 5),
        (0, 6) => (4, 5), (4, 5) => (0, 6),
        (1, 5) => (3, 3), (3, 3) => (1, 5),
        (1, 6) => (4, 3), (4, 3) => (1, 6),
        (2, 5) => (3, 4), (3, 4) => (2, 5),
        (2, 6) => (4, 4), (4, 4) => (2, 6)
    )
    
    connectivity_rows = [
        0 3 1 4; 1 4 0 3; 0 4 2 3; 2 3 0 4;
        0 5 3 6; 3 6 0 5; 0 6 4 5; 4 5 0 6;
        1 5 3 3; 3 3 1 5; 1 6 4 3; 4 3 1 6;
        2 5 3 4; 3 4 2 5; 2 6 4 4; 4 4 2 6
    ]
    
    function get_tangential_vector(blocks_data, bid, fid)
        x, y, z, Nx, Ny, Nz = blocks_data[bid + 1]
        mid_i = size(x, 1) ÷ 2
        if fid == 3      
            j0 = 1; mid_k = (Nz + 2) ÷ 2
            return [y[mid_i, j0, mid_k+1] - y[mid_i, j0, mid_k], 
                    z[mid_i, j0, mid_k+1] - z[mid_i, j0, mid_k]]
        elseif fid == 4  
            j0 = Ny + 1; mid_k = (Nz + 2) ÷ 2
            return [y[mid_i, j0, mid_k+1] - y[mid_i, j0, mid_k],
                    z[mid_i, j0, mid_k+1] - z[mid_i, j0, mid_k]]
        elseif fid == 5  
            k0 = 1; mid_j = (Ny + 2) ÷ 2
            return [y[mid_i, mid_j+1, k0] - y[mid_i, mid_j, k0],
                    z[mid_i, mid_j+1, k0] - z[mid_i, mid_j, k0]]
        elseif fid == 6  
            k0 = Nz + 1; mid_j = (Ny + 2) ÷ 2
            return [y[mid_i, mid_j+1, k0] - y[mid_i, mid_j, k0],
                    z[mid_i, mid_j+1, k0] - z[mid_i, mid_j, k0]]
        end
    end
    
    reverse_tan_arr = zeros(Int64, size(connectivity_rows, 1))
    flip_normal_arr = zeros(Int64, size(connectivity_rows, 1))
    for i in 1:size(connectivity_rows, 1)
        b1, f1, b2, f2 = connectivity_rows[i, :]
        t1 = get_tangential_vector(blocks_data, b1, f1)
        t2 = get_tangential_vector(blocks_data, b2, f2)
        reverse_tan_arr[i] = dot(t1, t2) < 0 ? 1 : 0
        
        f1_is_eta = (f1 == 3 || f1 == 4)
        f1_is_zeta = (f1 == 5 || f1 == 6)
        f2_is_eta = (f2 == 3 || f2 == 4)
        f2_is_zeta = (f2 == 5 || f2 == 6)
        if (f1_is_eta && f2_is_zeta) || (f1_is_zeta && f2_is_eta)
            flip_normal_arr[i] = 1
        end
    end
    axis_map = structured_connectivity_axis_map(connectivity_rows, reverse_tan_arr)
    
    face_bc = zeros(Int64, 5, 6) 
    for bid in 0:4
        face_bc[bid+1, 1] = 2  
        face_bc[bid+1, 2] = 2  
    end
    for i in 1:size(connectivity_rows, 1)
        b1, f1 = connectivity_rows[i, 1], connectivity_rows[i, 2]
        face_bc[b1+1, f1] = 0  
    end
    face_bc[2, 3] = 1  
    face_bc[3, 4] = 1  
    face_bc[4, 5] = 1  
    face_bc[5, 6] = 1  
    
    # BC parameters: store Tw for each isothermal wall face
    # BCP_TW = slot 1 (matching bc_types.jl)
    bc_params = zeros(FT, 5, 6, 17)  # 17 = N_BC_PARAMS (current version)
    bc_params[2, 3, 1] = FT(307.0e0)  # Block 1, η-, Tw
    bc_params[3, 4, 1] = FT(307.0e0)  # Block 2, η+, Tw
    bc_params[4, 5, 1] = FT(307.0e0)  # Block 3, ζ-, Tw
    bc_params[5, 6, 1] = FT(307.0e0)  # Block 4, ζ+, Tw
    
    h5open(joinpath(out_dir, "block_connectivity.h5"), "w") do f
        f["Nblocks"] = 5
        f["Nx_b"] = fill(Int64(Nx_b), 5)
        f["Ny_b"] = Int64[Ny_b0, N_rad, N_rad, Ny_b0, Ny_b0]
        f["Nz_b"] = Int64[Nz_b0, Nz_b0, Nz_b0, N_rad, N_rad]
        f["connectivity"] = connectivity_rows
        f["reverse_tan"] = reverse_tan_arr
        f["flip_normal"] = flip_normal_arr
        f["axis_map"] = axis_map
        f["face_bc"] = face_bc
        f["bc_params"] = bc_params
    end
    if vis; export_mesh_vtk(blocks_data, out_dir); end
    println("Done '", out_dir, "'!")
end

function main()
    build_mesh("MESH_COARSE", 512, 108, 108, 108)  # N_rad: 96 → 108
    build_mesh("MESH_FINE", 1024, 216, 216, 216)   # N_rad: 192 → 216
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
