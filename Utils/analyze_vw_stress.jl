# =============================================================================
#  analyze_vw_stress.jl  (memory-optimized)
#
#  后处理: 计算径向-周向 Reynolds 切应力 <v'_r v'_θ> 的径向分布
#  使用 AVG 平均场 + PLT 脉动, 体积加权 bin 平均
#
#  内存优化: 只保留 cos_theta, sin_theta, cell_area, bin_idx (3+1 数组/block)
#            yc, zc, rc 在初始化后释放 (~60% 内存节省)
#
#  用法:
#    julia Utils/analyze_vw_stress.jl PLT AVG/avg-1035000 [STEP_SPEC]
# =============================================================================

using HDF5
using Printf

const MESH_DIR = "MESH"
const NBLOCKS  = 5
const OUT_DIR  = "VW_STRESS"
const N_BINS   = 120
const SKIP_BND = 4
const R_MIN_FRAC = 0.02

# =============================================================================
#  工具函数
# =============================================================================

function find_complete_steps(dir::String, nblocks::Int)
    steps = Int[]
    for f in readdir(dir)
        m = match(r"plt-(\d+)-b0\.h5", f)
        if m !== nothing
            step = parse(Int, m.captures[1])
            all_exist = all(isfile(joinpath(dir, "plt-$(step)-b$(bid).h5")) for bid in 0:nblocks-1)
            all_exist && push!(steps, step)
        end
    end
    return sort(steps)
end

# =============================================================================
#  紧凑 Block 数据: 只保留计算需要的最小集
# =============================================================================

struct BlockData
    id::Int
    Nx::Int; Ny::Int; Nz::Int
    cos_theta::Array{Float64, 3}
    sin_theta::Array{Float64, 3}
    cell_area::Array{Float64, 3}
    bin_idx::Array{Int32, 3}
    v_mean::Array{Float64, 3}     # from AVG
    w_mean::Array{Float64, 3}     # from AVG
end

# =============================================================================
#  一次性加载: 网格 + AVG + bin映射, 中间数据立即释放
# =============================================================================

function load_block_all(bid::Int, avg_prefix::String,
                        R_wall::Float64, r_min::Float64, dr::Float64, n_bins::Int)
    # ── 网格节点 ──
    mpath = joinpath(MESH_DIR, "mesh_b$(bid).h5")
    y_nodes = h5read(mpath, "y")::Array{Float32, 3}
    z_nodes = h5read(mpath, "z")::Array{Float32, 3}

    # ── cell 中心 (临时, 用完释放) ──
    yc = 0.5 .* (Float64.(y_nodes[1:end-1,:,:]) .+ Float64.(y_nodes[2:end,:,:]))
    yc = 0.5 .* (yc[:,1:end-1,:] .+ yc[:,2:end,:])
    yc = 0.5 .* (yc[:,:,1:end-1] .+ yc[:,:,2:end])

    zc = 0.5 .* (Float64.(z_nodes[1:end-1,:,:]) .+ Float64.(z_nodes[2:end,:,:]))
    zc = 0.5 .* (zc[:,1:end-1,:] .+ zc[:,2:end,:])
    zc = 0.5 .* (zc[:,:,1:end-1] .+ zc[:,:,2:end])

    Nx, Ny, Nz = size(yc)

    # ── cos_theta, sin_theta, bin_idx (保留) ──
    cos_theta = Array{Float64}(undef, Nx, Ny, Nz)
    sin_theta = Array{Float64}(undef, Nx, Ny, Nz)
    bin_idx   = zeros(Int32, Nx, Ny, Nz)

    j_lo = 1 + SKIP_BND;  j_hi = Ny - SKIP_BND
    k_lo = 1 + SKIP_BND;  k_hi = Nz - SKIP_BND

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        y = yc[i,j,k]; z = zc[i,j,k]
        r = sqrt(y*y + z*z)
        if r < 1.0e-12
            cos_theta[i,j,k] = 0.0
            sin_theta[i,j,k] = 0.0
        else
            cos_theta[i,j,k] = y / r
            sin_theta[i,j,k] = z / r
        end
        # bin mapping
        if j >= j_lo && j <= j_hi && k >= k_lo && k <= k_hi && r >= r_min
            bin_idx[i,j,k] = clamp(Int32(floor((r - r_min) / dr)) + Int32(1), Int32(1), Int32(n_bins))
        end
    end

    # 释放 yc, zc (不再需要)
    yc = nothing; zc = nothing
    GC.gc()

    # ── cell_area (直接用 Float32 节点, 不做 Float64 拷贝) ──
    cell_area = Array{Float64}(undef, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        y1 = Float64(y_nodes[i, j,   k]);   z1 = Float64(z_nodes[i, j,   k])
        y2 = Float64(y_nodes[i, j+1, k]);   z2 = Float64(z_nodes[i, j+1, k])
        y3 = Float64(y_nodes[i, j+1, k+1]); z3 = Float64(z_nodes[i, j+1, k+1])
        y4 = Float64(y_nodes[i, j,   k+1]); z4 = Float64(z_nodes[i, j,   k+1])
        cell_area[i,j,k] = 0.5 * abs(
            (y1*z2 - y2*z1) + (y2*z3 - y3*z2) +
            (y3*z4 - y4*z3) + (y4*z1 - y1*z4)
        )
    end

    # 释放节点数组
    y_nodes = nothing; z_nodes = nothing
    GC.gc()

    # ── AVG 平均场 (只读 v, w 列) ──
    if avg_prefix == "NONE"
        v_mean = zeros(Float64, Nx, Ny, Nz)
        w_mean = zeros(Float64, Nx, Ny, Nz)
    else
        avg_path = "$(avg_prefix)-b$(bid).h5"
        v_mean, w_mean = h5open(avg_path, "r") do f
            avg = f["avg"]
            # 只读取 column 3 (v) 和 4 (w), 避免加载全部 6 列
            vm = Float64.(read(avg)[:, :, :, 3])
            wm = Float64.(read(avg)[:, :, :, 4])
            (vm, wm)
        end
    end
    GC.gc()

    @printf("  Block %d: (%d,%d,%d)  area∈[%.2e,%.2e]  ṽ∈[%.2f,%.2f]  w̃∈[%.2f,%.2f]\n",
            bid, Nx, Ny, Nz,
            minimum(cell_area), maximum(cell_area),
            minimum(v_mean), maximum(v_mean),
            minimum(w_mean), maximum(w_mean))

    return BlockData(bid, Nx, Ny, Nz, cos_theta, sin_theta, cell_area, bin_idx, v_mean, w_mean)
end

# =============================================================================
#  单步: 逐 block 加载 PLT → 计算脉动 → 累积 (每个 block 处理完立即释放)
# =============================================================================

function accumulate_step!(sum_vrvt::Vector{Float64},
                          sum_vr_fluc::Vector{Float64},
                          sum_vt_fluc::Vector{Float64},
                          sum_weight::Vector{Float64},
                          plt_dir::String, step::Int,
                          block_data::Vector{BlockData})
    for bid in 0:NBLOCKS-1
        fpath = joinpath(plt_dir, "plt-$(step)-b$(bid).h5")
        v_inst, w_inst = h5open(fpath, "r") do f
            Float64.(read(f["v"])), Float64.(read(f["w"]))
        end

        bd = block_data[bid+1]

        @inbounds for k in 1:bd.Nz, j in 1:bd.Ny, i in 1:bd.Nx
            b = bd.bin_idx[i,j,k]
            b == 0 && continue

            w_vol = bd.cell_area[i,j,k]
            v_f = v_inst[i,j,k] - bd.v_mean[i,j,k]
            w_f = w_inst[i,j,k] - bd.w_mean[i,j,k]

            cosθ = bd.cos_theta[i,j,k]
            sinθ = bd.sin_theta[i,j,k]
            vr_f =  v_f * cosθ + w_f * sinθ
            vt_f = -v_f * sinθ + w_f * cosθ

            sum_vrvt[b]    += w_vol * vr_f * vt_f
            sum_vr_fluc[b] += w_vol * vr_f
            sum_vt_fluc[b] += w_vol * vt_f
            sum_weight[b]  += w_vol
        end

        # 立即释放当前 block 的 PLT 数据
        v_inst = nothing; w_inst = nothing
    end
    GC.gc()
end

# =============================================================================
#  MAIN
# =============================================================================

println("=" ^ 70)
println("  Reynolds Stress ⟨v'_r v'_θ⟩ — Memory-Optimized")
println("=" ^ 70)

if length(ARGS) < 2
    println("用法: julia Utils/analyze_vw_stress.jl PLT_DIR AVG_PREFIX [STEP_SPEC]")
    println("  例: julia Utils/analyze_vw_stress.jl PLT AVG/avg-1035000 841000:1035000")
    exit(1)
end

plt_dir    = ARGS[1]
avg_prefix = ARGS[2]

all_steps = find_complete_steps(plt_dir, NBLOCKS)
@printf("  Found %d complete PLT timesteps in %s\n", length(all_steps), plt_dir)

if length(ARGS) >= 3
    spec = ARGS[3]
    parts = split(spec, ":")
    if length(parts) == 2
        s1, s2 = parse(Int, parts[1]), parse(Int, parts[2])
        all_steps = filter(s -> s1 <= s <= s2, all_steps)
    elseif length(parts) == 3
        s1, interval, s2 = parse(Int, parts[1]), parse(Int, parts[2]), parse(Int, parts[3])
        candidates = filter(s -> s1 <= s <= s2, all_steps)
        if length(candidates) > 0
            sampled = Int[candidates[1]]
            for s in candidates[2:end-1]
                (s - sampled[end] >= interval) && push!(sampled, s)
            end
            push!(sampled, candidates[end])
            all_steps = unique(sampled)
        end
    end
end

isempty(all_steps) && error("No valid PLT steps found!")
@printf("  Processing %d steps: [%d → %d]\n\n", length(all_steps), all_steps[1], all_steps[end])

mkpath(OUT_DIR)

# ── 预计算 bin 参数 ──
# 先快速扫描一遍 mesh 得到 R_wall
R_wall = 0.0
for bid in 0:NBLOCKS-1
    global R_wall
    mpath = joinpath(MESH_DIR, "mesh_b$(bid).h5")
    y = h5read(mpath, "y")::Array{Float32, 3}
    z = h5read(mpath, "z")::Array{Float32, 3}
    for idx in eachindex(y)
        r = sqrt(Float64(y[idx])^2 + Float64(z[idx])^2)
        R_wall = max(R_wall, r)
    end
end
r_min = R_MIN_FRAC * R_wall
dr = (R_wall - r_min) / N_BINS
r_R = [(r_min + (i - 0.5) * dr) / R_wall for i in 1:N_BINS]
GC.gc()
@printf("  R_wall = %.4f, %d bins, r/R ∈ [%.3f, %.3f]\n\n", R_wall, N_BINS, r_R[1], r_R[end])

# ── 加载所有 block (网格 + AVG + bin) ──
println("── Loading blocks (mesh + AVG + bins) ──")
block_data = [load_block_all(bid, avg_prefix, R_wall, r_min, dr, N_BINS) for bid in 0:NBLOCKS-1]
GC.gc()

# ── 输出均值径向 profile ──
println("\n── Mean velocity profile ──")
vr_mean_bin = zeros(N_BINS)
vt_mean_bin = zeros(N_BINS)
wt_bin      = zeros(N_BINS)

for bd in block_data
    @inbounds for k in 1:bd.Nz, j in 1:bd.Ny, i in 1:bd.Nx
        b = bd.bin_idx[i,j,k]; b == 0 && continue
        w_vol = bd.cell_area[i,j,k]
        cosθ = bd.cos_theta[i,j,k]; sinθ = bd.sin_theta[i,j,k]
        vr =  bd.v_mean[i,j,k] * cosθ + bd.w_mean[i,j,k] * sinθ
        vt = -bd.v_mean[i,j,k] * sinθ + bd.w_mean[i,j,k] * cosθ
        vr_mean_bin[b] += w_vol * vr; vt_mean_bin[b] += w_vol * vt; wt_bin[b] += w_vol
    end
end
for i in 1:N_BINS
    if wt_bin[i] > 0; vr_mean_bin[i] /= wt_bin[i]; vt_mean_bin[i] /= wt_bin[i]; end
end
open(joinpath(OUT_DIR, "mean_velocity_profile.csv"), "w") do io
    println(io, "r_R,vr_mean,vt_mean")
    for i in 1:N_BINS
        @printf(io, "%.8e,%.8e,%.8e\n", r_R[i], vr_mean_bin[i], vt_mean_bin[i])
    end
end
@printf("  ⟨v_r⟩_rms = %.4e  ⟨v_θ⟩ range = [%.4f, %.4f]\n",
        sqrt(sum(vr_mean_bin .^ 2) / N_BINS), minimum(vt_mean_bin), maximum(vt_mean_bin))

# ── 累积 Reynolds 应力 ──
println("\n" * "=" ^ 50)
println("  Computing ⟨v'_r v'_θ⟩")
println("=" ^ 50)

sum_vrvt    = zeros(N_BINS)
sum_vr_fluc = zeros(N_BINS)
sum_vt_fluc = zeros(N_BINS)
sum_weight  = zeros(N_BINS)

io_re   = open(joinpath(OUT_DIR, "vw_reynolds_stress.csv"), "w")
io_mean = open(joinpath(OUT_DIR, "vw_cumulative_mean.csv"), "w")
io_conv = open(joinpath(OUT_DIR, "vw_convergence.csv"), "w")

println(io_re,   "n_samples,step,r_R,vw_reynolds")
println(io_mean, "n_samples,step,r_R,vrvt_fluc,vr_fluc_mean,vt_fluc_mean")
println(io_conv, "n_samples,step,delta_rms,delta_max,vw_re_rms")

prev_re = nothing

for (tidx, step) in enumerate(all_steps)
    global prev_re
    t0 = time()

    accumulate_step!(sum_vrvt, sum_vr_fluc, sum_vt_fluc, sum_weight,
                     plt_dir, step, block_data)

    re_stress = zeros(N_BINS)
    vr_f_mean = zeros(N_BINS)
    vt_f_mean = zeros(N_BINS)

    for i in 1:N_BINS
        if sum_weight[i] > 0
            vr_m = sum_vr_fluc[i] / sum_weight[i]
            vt_m = sum_vt_fluc[i] / sum_weight[i]
            vr_f_mean[i] = vr_m
            vt_f_mean[i] = vt_m
            re_stress[i] = (sum_vrvt[i] / sum_weight[i]) - (vr_m * vt_m)
        end
    end

    delta_rms = 0.0; delta_max = 0.0
    if prev_re !== nothing
        diff = re_stress .- prev_re
        delta_rms = sqrt(sum(diff .^ 2) / N_BINS)
        delta_max = maximum(abs.(diff))
    end
    re_rms = sqrt(sum(re_stress .^ 2) / N_BINS)
    prev_re = copy(re_stress)

    elapsed = time() - t0
    vr_res = sqrt(sum(vr_f_mean .^ 2) / N_BINS)

    @printf("  [%3d/%d] Step %8d  Δrms=%.3e  |Re|=%.3e  ⟨v'⟩=%.2e  (%.1f s)\n",
            tidx, length(all_steps), step, delta_rms, re_rms, vr_res, elapsed)

    for i in 1:N_BINS
        @printf(io_re,   "%d,%d,%.8e,%.8e\n", tidx, step, r_R[i], re_stress[i])
        @printf(io_mean, "%d,%d,%.8e,%.8e,%.8e,%.8e\n", tidx, step, r_R[i],
                re_stress[i], vr_f_mean[i], vt_f_mean[i])
    end
    @printf(io_conv, "%d,%d,%.8e,%.8e,%.8e\n", tidx, step, delta_rms, delta_max, re_rms)

    flush(io_re); flush(io_mean); flush(io_conv)
end

close(io_re); close(io_mean); close(io_conv)

println("\n" * "=" ^ 70)
println("  Done!")
@printf("  Steps: %d [%d → %d]\n", length(all_steps), all_steps[1], all_steps[end])
println("  Output: $(OUT_DIR)/")
println("=" ^ 70)
