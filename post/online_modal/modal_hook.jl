# =============================================================================
#  modal_hook.jl - 在线模态分解钩子（in_situ_post_process）
#
#  由 run 脚本 include 后，在 Main 作用域定义 in_situ_post_process。
#  求解器在 solver.jl:3527 通过 isdefined(Main, :in_situ_post_process) 调用。
#
#  触发时机：RK3 完成后、activeTime += current_dt 之前。
#  此时 b.Q 是新解 U^{n+1}；时间戳用 activeTime + current_dt。
#
#  每个采样步：
#    1. 拷 5 块速度切片到 CPU（仅 interior，剥离 NG）
#    2. 重采样到圆柱网格 (r, θ, x)
#    3. 周向 FFT -> Fourier 系数 û_m(r, x, t)
#    4. 速度分量变换 (u,v,w)->(u_x,u_r,u_θ)
#    5. 落盘 HDF5：u_hat[m,Nr,Nx,3] + 时间戳
#
#  磁盘占用：~N_m×Nr×Nx×3 复数 ×16B ≈ 几十 MB/快照（vs 3.2GB 全场）
# =============================================================================

using HDF5
using FFTW
using Printf

# =============================================================================
#  全局缓存：映射表（只构建一次）
# =============================================================================
const _modal_state = Dict{Symbol, Any}(
    :rmap => nothing,
    :initialized => false,
    :sample_count => 0,
)

"""
    _ensure_initialized(blocks)

首次调用时预计算重采样映射表。从 mesh_dir 读取网格坐标。
mesh_dir 由 run 脚本定义为 const（见 pipe_caseB.jl:95）。
"""
function _ensure_initialized(blocks)
    if _modal_state[:initialized]
        return
    end

    # 获取 mesh_dir：run 脚本中定义的 const
    mesh_dir = Main.mesh_dir
    @printf("[modal] 初始化重采样映射表 (mesh_dir=%s)...\n", mesh_dir)
    t0 = time_ns()
    _modal_state[:rmap] = build_regrid_map(mesh_dir)
    t_init = (time_ns() - t0) / 1e9
    rmap = _modal_state[:rmap]
    @printf("[modal] 映射表就绪 (%.2fs): nx=%d ntheta=%d nrad=%d\n",
            t_init, rmap.nx, rmap.ntheta, rmap.nrad)

    # 创建输出目录
    mkpath(modal_out_dir)

    _modal_state[:initialized] = true
end


# =============================================================================
#  主钩子
# =============================================================================
"""
    in_situ_post_process(tt, activeTime, current_dt, blocks,
                         world_rank, Block_Nprocs, block_comms)

在线模态分解钩子。每 modal_sample_step 步触发一次。
"""
function in_situ_post_process(tt, activeTime, current_dt, blocks,
                              world_rank, Block_Nprocs, block_comms)
    # 仅在采样步触发
    if tt % modal_sample_step != 0
        return
    end

    _ensure_initialized(blocks)
    rmap = _modal_state[:rmap]

    # 时间戳：U^{n+1} 对应 activeTime + current_dt
    # （钩子在 activeTime += current_dt 之前触发，b.Q 是新解）
    t_stamp = activeTime + current_dt

    # ── 1. 拷 5 块速度切片到 CPU ──
    # blocks 是 Dict{Int, Block}，b.Q 是 GPUArray{FT,4} = [ρ,u,v,w,p,T,...]
    # 速度分量 q=2,3,4。interior 范围 [NGp:nx_end]。
    NGp = NG + 1
    block_data = Dict{Int, NTuple{3, Array{Float64,3}}}()

    for (bid, b) in blocks
        if bid >= 5
            continue  # 只处理主管 5 块（跳过 CEBL precursor 块 5-9）
        end
        nx_end = b.Nx + NG
        ny_end = b.Ny + NG
        nz_end = b.Nz + NG

        # 拷 interior 速度切片到 CPU
        u_h = Array(@view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 2])
        v_h = Array(@view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 3])
        w_h = Array(@view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 4])

        block_data[bid] = (Float64.(u_h), Float64.(v_h), Float64.(w_h))
    end

    # 确保所有 5 块都在（多 GPU 分区下每块在不同 rank）
    # 当前实现只支持单 rank 持有全部主管块。
    if length(block_data) < 5
        world_rank == 0 && @warn(
            "[modal] 跳过在线模态采样：当前实现要求单 rank 持有全部 5 个主管块",
            local_blocks=sort!(collect(keys(block_data))),
        )
        return
    end

    # ── 2. 重采样到圆柱网格 ──
    u_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
    v_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
    w_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
    regrid_to_cyl!(u_cyl, v_cyl, w_cyl, rmap, block_data)

    # ── 3. 周向 FFT + 速度分量变换 ──
    # 对每个 (ix, ir) 做沿 θ 的 FFT。
    # 速度变换：笛卡尔 (u,v,w)=(ux,uy,uz) -> 圆柱 (ux, ur, utheta)
    #   ur    =  uy*cos(θ) + uz*sin(θ)
    #   utheta= -uy*sin(θ) + uz*cos(θ)
    # 在频域：θ 旋转是卷积，但在实空间做更简单。
    # 先在实空间变换分量，再 FFT。
    ntheta = rmap.ntheta
    ntheta_half = ntheta ÷ 2 + 1  # rfft 输出长度
    nmodes = min(modal_nmodes_keep, ntheta_half)

    # 实空间分量变换
    ur_cyl    = zeros(rmap.nx, ntheta, rmap.nrad)
    uth_cyl   = zeros(rmap.nx, ntheta, rmap.nrad)
    for ir in 1:rmap.nrad, it in 1:ntheta
        θ = rmap.theta_grid[it]
        c = cos(θ); s = sin(θ)
        @inbounds for ix in 1:rmap.nx
            uy = v_cyl[ix, it, ir]
            uz = w_cyl[ix, it, ir]
            ur_cyl[ix, it, ir]  =  uy * c + uz * s
            uth_cyl[ix, it, ir] = -uy * s + uz * c
        end
    end

    # 周向 FFT（rfft，实输入->复数）
    # 输出: u_hat[nmodes, Nr, Nx] per component（m=0..nmodes-1）
    u_hat  = zeros(ComplexF64, nmodes, rmap.nrad, rmap.nx)
    ur_hat = zeros(ComplexF64, nmodes, rmap.nrad, rmap.nx)
    uth_hat= zeros(ComplexF64, nmodes, rmap.nrad, rmap.nx)

    for ix in 1:rmap.nx, ir in 1:rmap.nrad
        # r=0 附近 m>=1 强制 0（物理约束：轴心无周向结构）
        if rmap.r_grid[ir] < r_axis_cut
            u_hat[1, ir, ix]  = sum(@view u_cyl[ix,:,ir]) / ntheta  # m=0 取平均
            ur_hat[1, ir, ix] = sum(@view ur_cyl[ix,:,ir]) / ntheta
            uth_hat[1, ir, ix]= sum(@view uth_cyl[ix,:,ir]) / ntheta
            continue
        end
        fu  = FFTW.rfft(view(u_cyl,  ix, :, ir))
        fur = FFTW.rfft(view(ur_cyl, ix, :, ir))
        futh= FFTW.rfft(view(uth_cyl,ix, :, ir))
        for m in 1:nmodes
            u_hat[m, ir, ix]   = fu[m]
            ur_hat[m, ir, ix]  = fur[m]
            uth_hat[m, ir, ix] = futh[m]
        end
    end

    # ── 4. 落盘（仅 rank 0）──
    if world_rank == 0
        outpath = joinpath(modal_out_dir, "coeff-$(tt).h5")
        h5open(outpath, "w") do f
            # 时间戳（标量属性）
            attrs(f)["time"] = Float64(t_stamp)
            attrs(f)["step"] = Int(tt)
            attrs(f)["dt"]   = Float64(current_dt)

            # 网格信息
            f["r_grid"]    = collect(rmap.r_grid)
            f["theta_grid"]= collect(rmap.theta_grid)
            f["x_grid"]    = collect(rmap.x_grid)

            # Fourier 系数 [m, Nr, Nx]
            f["u_hat"]   = u_hat
            f["ur_hat"]  = ur_hat
            f["uth_hat"] = uth_hat
        end
        _modal_state[:sample_count] += 1
        if _modal_state[:sample_count] % 10 == 1
            @printf("[modal] step=%d t=%.6e 已写 %s (累计 %d 快照)\n",
                    tt, t_stamp, outpath, _modal_state[:sample_count])
        end
    end

    return nothing
end

# =============================================================================
#  注：多 GPU 分区支持
#  auto_partition 下，5 块可能分布在多个 rank。当前实现假设单 rank 持有全部块。
#  完整多 rank 支持需要：
#    1. 每个 rank 对本地块做重采样（部分圆柱网格点）
#    2. MPI_Allgatherv 收集到 rank 0 的完整圆柱网格
#    3. rank 0 做 FFT + 落盘
#  这部分留待生产部署时实现。本地测试（单 GPU/全部块在一个 rank）可直接用。
# =============================================================================
