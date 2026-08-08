function apply_structured_split_sources!(blocks,source_dt::FT,launch_threads)
    source_dt>zero(FT) || return false
    applied=false
    for (_,block) in blocks
        interior_blocks=(
            cld(block.Nx,launch_threads[1]),cld(block.Ny,launch_threads[2]),
            cld(block.Nz,launch_threads[3]))
        if equation_type==:MHD && !ct_mode
            @gpu_launch threads=launch_threads blocks=interior_blocks glm_source_kernel!(
                block.U,block.Q,source_dt,ch_glm_current,cr_glm,
                block.Nx,block.Ny,block.Nz)
            applied=true
        end
        if _outlet_sponge_active && block.sponge_sigma!==nothing
            full_blocks=(
                cld(block.Nx+2NG,launch_threads[1]),
                cld(block.Ny+2NG,launch_threads[2]),
                cld(block.Nz+2NG,launch_threads[3]))
            @gpu_launch threads=launch_threads blocks=full_blocks sponge_step_kernel!(
                block.U,block.sponge_U_target,block.sponge_sigma,
                source_dt,block.Nx,block.Ny,block.Nz)
            @gpu_launch threads=launch_threads blocks=full_blocks c2Prim(
                block.U,block.Q,block.Nx,block.Ny,block.Nz)
            applied=true
        end
    end
    return applied
end
