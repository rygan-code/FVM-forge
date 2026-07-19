# =============================================================================
# analyze_steady_state.jl - 增量统计,低内存,判断统计稳态
#
# 策略:逐 PLT 读取,每个只提取 x≈x_target 的截面(1 个 i 切片),
# 累加到统计量中,然后释放。内存峰值 = 1 个 PLT 的大小。
# 比较后半段两组样本的剖面 -> 吻合则统计稳态。
# =============================================================================
using HDF5
using Printf

const R0 = 0.5
const MESH_DIR = abspath(joinpath(@__DIR__, "MESH_LEN15"))
const TMP_RUNS = abspath(joinpath(@__DIR__, "tmp_runs"))

# cell-center 坐标缓存(每个 block 只算一次)
const _xc_cache = Dict{Int, Array{Float64,3}}()
const _yc_cache = Dict{Int, Array{Float64,3}}()
const _zc_cache = Dict{Int, Array{Float64,3}}()

function get_cellcenter_coords(bid)
    haskey(_xc_cache, bid) && return _xc_cache[bid], _yc_cache[bid], _zc_cache[bid]
    coords = h5open(joinpath(MESH_DIR, "mesh_b$(bid).h5"), "r") do f; read(f["coords"]); end
    xn = Float64.(coords[1,:,:,:]); yn = Float64.(coords[2,:,:,:]); zn = Float64.(coords[3,:,:,:])
    x = @views 0.125 .* (xn[1:end-1,1:end-1,1:end-1].+xn[2:end,1:end-1,1:end-1].+xn[1:end-1,2:end,1:end-1].+xn[2:end,2:end,1:end-1].+xn[1:end-1,1:end-1,2:end].+xn[2:end,1:end-1,2:end].+xn[1:end-1,2:end,2:end].+xn[2:end,2:end,2:end])
    y = @views 0.125 .* (yn[1:end-1,1:end-1,1:end-1].+yn[2:end,1:end-1,1:end-1].+yn[1:end-1,2:end,1:end-1].+yn[2:end,2:end,1:end-1].+yn[1:end-1,1:end-1,2:end].+yn[2:end,1:end-1,2:end].+yn[1:end-1,2:end,2:end].+yn[2:end,2:end,2:end])
    z = @views 0.125 .* (zn[1:end-1,1:end-1,1:end-1].+zn[2:end,1:end-1,1:end-1].+zn[1:end-1,2:end,1:end-1].+zn[2:end,2:end,1:end-1].+zn[1:end-1,1:end-1,2:end].+zn[2:end,1:end-1,2:end].+zn[1:end-1,2:end,2:end].+zn[2:end,2:end,2:end])
    _xc_cache[bid] = x; _yc_cache[bid] = y; _zc_cache[bid] = z
    return x, y, z
end

"""找最接近 x_target 的 i 索引(对每个 block)"""
function find_i_target(bid, x_target)
    x, _, _ = get_cellcenter_coords(bid)
    nx = size(x, 1)
    # 用中段 j,k 评估 x 值
    best_i, best_d = 1, Inf
    for i in 1:nx
        d = abs(x[i, size(x,2)÷2, size(x,3)÷2] - x_target)
        d < best_d && (best_d = d; best_i = i)
    end
    return best_i
end

"""增量统计累加器:按 r 的 bin 累加 u,v,w 及其乘积"""
mutable struct BinAccumulator
    n_bins::Int
    cnt::Vector{Int}
    sum_r::Vector{Float64}
    sum_u::Vector{Float64}; sum_v::Vector{Float64}; sum_w::Vector{Float64}
    sum_uu::Vector{Float64}; sum_vv::Vector{Float64}; sum_ww::Vector{Float64}
    sum_uv::Vector{Float64}; sum_uw::Vector{Float64}; sum_vw::Vector{Float64}
end

BinAccumulator(n_bins) = BinAccumulator(n_bins,
    zeros(Int,n_bins), zeros(n_bins),
    zeros(n_bins), zeros(n_bins), zeros(n_bins),
    zeros(n_bins), zeros(n_bins), zeros(n_bins),
    zeros(n_bins), zeros(n_bins), zeros(n_bins))

function accumulate!(acc, r, u, v, w)
    for idx in eachindex(r)
        ri = r[idx]
        ri > R0 && continue
        bi = min(floor(Int, ri / R0 * acc.n_bins) + 1, acc.n_bins)
        bi < 1 && (bi = 1)
        ui, vi, wi = u[idx], v[idx], w[idx]
        acc.cnt[bi] += 1
        acc.sum_r[bi] += ri
        acc.sum_u[bi] += ui; acc.sum_v[bi] += vi; acc.sum_w[bi] += wi
        acc.sum_uu[bi] += ui*ui; acc.sum_vv[bi] += vi*vi; acc.sum_ww[bi] += wi*wi
        acc.sum_uv[bi] += ui*vi; acc.sum_uw[bi] += ui*wi; acc.sum_vw[bi] += vi*wi
    end
end

function finalize(acc)
    r_out=Float64[]; u=Float64[]; uv=Float64[]; uu=Float64[]
    for bi in 1:acc.n_bins
        acc.cnt[bi]==0 && continue
        c = acc.cnt[bi]
        um = acc.sum_u[bi]/c
        push!(r_out, acc.sum_r[bi]/c)
        push!(u, um)
        push!(uv, acc.sum_uv[bi]/c - um*(acc.sum_v[bi]/c))
        push!(uu, acc.sum_uu[bi]/c - um*um)
    end
    return (r=r_out, u=u, uv=uv, uu=uu)
end

function list_plt_steps(plt_dir)
    steps = Int[]
    isdir(plt_dir) || return steps
    for f in readdir(plt_dir)
        m = match(r"plt-(\d+)-b0\.h5$", f)
        m !== nothing && push!(steps, parse(Int, m.captures[1]))
    end
    return sort(steps)
end

function analyze_case(name, dirname, x_target; n_bins=40)
    println("\n" * "=" ^ 70)
    println("  $name @ x=$x_target")
    println("=" ^ 70)
    plt_dir = joinpath(TMP_RUNS, dirname, "PLT")
    steps = list_plt_steps(plt_dir)
    n = length(steps)
    n == 0 && (println("  NO PLT"); return)
    println("  PLT steps: $(steps[1])..$(steps[end]) ($n total)")

    # 两组:后半段末尾 vs 中段
    n_samp = min(8, n÷2)
    late_steps = steps[end-n_samp+1:end]
    early_steps = steps[max(1,end-2*n_samp+1):max(1,end-n_samp)]
    length(early_steps) < 3 && (early_steps = steps[1:n_samp])
    println("  Early: $(early_steps[1])..$(early_steps[end]) ($(length(early_steps)) samples)")
    println("  Late:  $(late_steps[1])..$(late_steps[end]) ($(length(late_steps)) samples)")

    # 预算每个 block 的 i_target
    i_targets = [find_i_target(b, x_target) for b in 0:4]

    function run_accumulator(step_list)
        acc = BinAccumulator(n_bins)
        for s in step_list
            for bid in 0:4
                fname = joinpath(plt_dir, "plt-$(s)-b$(bid).h5")
                isfile(fname) || continue
                xc, yc, zc = get_cellcenter_coords(bid)
                it = i_targets[bid+1]
                u_slice = v_slice = w_slice = nothing
                h5open(fname, "r") do f
                    u_slice = read(f["u"])[it, :, :]
                    v_slice = read(f["v"])[it, :, :]
                    w_slice = read(f["w"])[it, :, :]
                end
                # 提取截面 r, u, v, w
                ny, nz = size(u_slice)
                r_arr = zeros(ny*nz); u_arr = zeros(ny*nz)
                v_arr = zeros(ny*nz); w_arr = zeros(ny*nz)
                idx = 0
                for k in 1:nz, j in 1:ny
                    ycj = yc[it,j,k]; zcj = zc[it,j,k]
                    r = sqrt(ycj^2 + zcj^2)
                    r > R0 && continue
                    idx += 1
                    r_arr[idx]=r; u_arr[idx]=u_slice[j,k]
                    v_arr[idx]=v_slice[j,k]; w_arr[idx]=w_slice[j,k]
                end
                resize!(r_arr, idx); resize!(u_arr, idx)
                resize!(v_arr, idx); resize!(w_arr, idx)
                accumulate!(acc, r_arr, u_arr, v_arr, w_arr)
            end
            println("    processed step $s")
            flush(stdout)
        end
        return finalize(acc)
    end

    println("  Accumulating early period...")
    se = run_accumulator(early_steps)
    println("  Accumulating late period...")
    sl = run_accumulator(late_steps)

    # 对比
    println("\n  r/R    | ⟨u⟩_early   ⟨u⟩_late    Δ%   | ⟨u'v'⟩_early  ⟨u'v'⟩_late  Δ%")
    println("  " * "-" ^ 85)
    nn = min(length(se.r), length(sl.r))
    max_du = 0.0; max_duv = 0.0
    for i in 1:nn
        ra = (se.r[i]+sl.r[i])/2/R0
        du = abs(se.u[i])>1e-10 ? abs(se.u[i]-sl.u[i])/abs(se.u[i])*100 : 0
        duv = abs(se.uv[i])>1e-10 ? abs(se.uv[i]-sl.uv[i])/abs(se.uv[i])*100 : 0
        max_du = max(max_du, du); max_duv = max(max_duv, duv)
        @printf("  %.3f  | %10.2f  %10.2f  %5.1f%% | %12.4f %12.4f %5.1f%%\n",
                ra, se.u[i], sl.u[i], du, se.uv[i], sl.uv[i], duv)
    end
    println("\n  --- 稳态判断 ---")
    @printf("  max|Δ⟨u⟩|/|⟨u⟩| = %.1f%%\n", max_du)
    @printf("  max|Δ⟨u'v'⟩|/|⟨u'v'⟩| = %.1f%%\n", max_duv)
    if max_du < 5.0 && max_duv < 15.0
        println("  => 统计稳态 ✓")
    else
        println("  => 尚未稳态 ✗")
    end
    flush(stdout)
end

const CASES = [
    ("Case A (差速旋转 Ro 0->1)", "caseA_diffrot"),
    ("Case B (均匀旋转 Ro=1.0)",  "caseB_uniform"),
    ("Case C (无旋转)",           "caseC_norot"),
]
const X_TARGET = 7.5

for (name, d) in CASES
    analyze_case(name, d, X_TARGET)
end
