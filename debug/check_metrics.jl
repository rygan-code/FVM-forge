# =============================================================================
# check_metrics.jl — Phase 3 Diagnostic: Metric Consistency at Block Interfaces
#
# Verifies that face normals and areas are consistent across block interfaces.
# For a shared face: Area_A should equal Area_B, and n_A should equal -n_B.
#
# Usage: include this file, then call
#        check_metric_consistency(blocks, connectivity, world_rank)
# =============================================================================

"""
    check_metric_consistency(blocks, connectivity, world_rank)

Check that face normals and areas are consistent across block interfaces.
"""
function check_metric_consistency(blocks, connectivity, world_rank)
    mkpath("debug")

    for (conn_key, conn) in connectivity
        dst_b_id, dst_fid = conn_key
        src_b_id = conn.src_b
        src_fid = conn.src_f

        if !haskey(blocks, dst_b_id) || !haskey(blocks, src_b_id)
            continue
        end

        dst_b = blocks[dst_b_id]
        src_b = blocks[src_b_id]

        nxp_d, nyp_d, nzp_d = dst_b.Nx, dst_b.Ny, dst_b.Nz
        nxp_s, nyp_s, nzp_s = src_b.Nx, src_b.Ny, src_b.Nz

        fname = "debug/metrics_b$(src_b_id)_f$(src_fid)_to_b$(dst_b_id)_f$(dst_fid)_rank$(world_rank).txt"
        open(fname, "w") do io
            println(io, "# Metric consistency check: Block $(src_b_id) face $(src_fid) → Block $(dst_b_id) face $(dst_fid)")
            println(io, "# Rank: $world_rank")
            println(io, "#")

            if dst_fid == 1 || dst_fid == 2  # ξ interface
                # Check at multiple j,k points
                for j_off in 0:2
                    for k_off in 0:2
                        j_d = div(nyp_d, 4) * (j_off + 1) + NG
                        k_d = div(nzp_d, 4) * (k_off + 1) + NG

                        if dst_fid == 1  # ξ-lo of dst
                            # Face index: i = NG+1 (node-indexed)
                            face_idx_dst = NG + 1
                            # Source ξ-hi: face index = nxp_s + NG + 1
                            face_idx_src = nxp_s + NG + 1
                        else  # ξ-hi of dst
                            face_idx_dst = nxp_d + NG + 1
                            face_idx_src = NG + 1
                        end

                        # Get metrics (node-indexed arrays)
                        Area_dst = Array(dst_b.Areai)[face_idx_dst, j_d, k_d]
                        nx_dst = Array(dst_b.nxi)[face_idx_dst, j_d, k_d]
                        ny_dst = Array(dst_b.nyi)[face_idx_dst, j_d, k_d]
                        nz_dst = Array(dst_b.nzi)[face_idx_dst, j_d, k_d]

                        # For source, map j,k indices (may need reverse_tan)
                        # Simplified: assume same j,k mapping for now
                        j_s = j_d  # TODO: handle reverse_tan
                        k_s = k_d

                        Area_src = Array(src_b.Areai)[face_idx_src, j_s, k_s]
                        nx_src = Array(src_b.nxi)[face_idx_src, j_s, k_s]
                        ny_src = Array(src_b.nyi)[face_idx_src, j_s, k_s]
                        nz_src = Array(src_b.nzi)[face_idx_src, j_s, k_s]

                        # Check: Area should match, normals should be opposite
                        Area_err = abs(Area_dst - Area_src) / max(Area_dst, 1.0e-30)
                        nx_err = abs(nx_dst + nx_src)  # should be opposite
                        ny_err = abs(ny_dst + ny_src)
                        nz_err = abs(nz_dst + nz_src)

                        if j_off == 0 && k_off == 0
                            println(io, "# j_d=$j_d, k_d=$k_d (first sample)")
                            println(io, "#   Area_dst=$Area_dst, Area_src=$Area_src, rel_err=$Area_err")
                            println(io, "#   n_dst=($nx_dst, $ny_dst, $nz_dst)")
                            println(io, "#   n_src=($nx_src, $ny_src, $nz_src)")
                            println(io, "#   normal_opposite_err=($nx_err, $ny_err, $nz_err)")
                        end

                        # Flag if errors are large
                        if Area_err > 1.0e-6 || nx_err > 1.0e-6 || ny_err > 1.0e-6 || nz_err > 1.0e-6
                            println(io, "# WARNING: Large mismatch at j=$j_d, k=$k_d")
                            println(io, "#   Area_err=$Area_err, normal_err=($nx_err, $ny_err, $nz_err)")
                        end
                    end
                end

            elseif dst_fid == 3 || dst_fid == 4  # η interface
                for i_off in 0:2
                    for k_off in 0:2
                        i_d = div(nxp_d, 4) * (i_off + 1) + NG
                        k_d = div(nzp_d, 4) * (k_off + 1) + NG

                        if dst_fid == 3
                            face_idx_dst = NG + 1
                            face_idx_src = nyp_s + NG + 1
                        else
                            face_idx_dst = nyp_d + NG + 1
                            face_idx_src = NG + 1
                        end

                        Area_dst = Array(dst_b.Areaj)[i_d, face_idx_dst, k_d]
                        nx_dst = Array(dst_b.nxj)[i_d, face_idx_dst, k_d]
                        ny_dst = Array(dst_b.nyj)[i_d, face_idx_dst, k_d]
                        nz_dst = Array(dst_b.nzj)[i_d, face_idx_dst, k_d]

                        i_s = i_d
                        k_s = k_d

                        Area_src = Array(src_b.Areaj)[i_s, face_idx_src, k_s]
                        nx_src = Array(src_b.nxj)[i_s, face_idx_src, k_s]
                        ny_src = Array(src_b.nyj)[i_s, face_idx_src, k_s]
                        nz_src = Array(src_b.nzj)[i_s, face_idx_src, k_s]

                        Area_err = abs(Area_dst - Area_src) / max(Area_dst, 1.0e-30)
                        nx_err = abs(nx_dst + nx_src)
                        ny_err = abs(ny_dst + ny_src)
                        nz_err = abs(nz_dst + nz_src)

                        if i_off == 0 && k_off == 0
                            println(io, "# i_d=$i_d, k_d=$k_d (first sample)")
                            println(io, "#   Area_dst=$Area_dst, Area_src=$Area_src, rel_err=$Area_err")
                            println(io, "#   n_dst=($nx_dst, $ny_dst, $nz_dst)")
                            println(io, "#   n_src=($nx_src, $ny_src, $nz_src)")
                            println(io, "#   normal_opposite_err=($nx_err, $ny_err, $nz_err)")
                        end

                        if Area_err > 1.0e-6 || nx_err > 1.0e-6 || ny_err > 1.0e-6 || nz_err > 1.0e-6
                            println(io, "# WARNING: Large mismatch at i=$i_d, k=$k_d")
                            println(io, "#   Area_err=$Area_err, normal_err=($nx_err, $ny_err, $nz_err)")
                        end
                    end
                end

            elseif dst_fid == 5 || dst_fid == 6  # ζ interface
                for i_off in 0:2
                    for j_off in 0:2
                        i_d = div(nxp_d, 4) * (i_off + 1) + NG
                        j_d = div(nyp_d, 4) * (j_off + 1) + NG

                        if dst_fid == 5
                            face_idx_dst = NG + 1
                            face_idx_src = nzp_s + NG + 1
                        else
                            face_idx_dst = nzp_d + NG + 1
                            face_idx_src = NG + 1
                        end

                        Area_dst = Array(dst_b.Areak)[i_d, j_d, face_idx_dst]
                        nx_dst = Array(dst_b.nxk)[i_d, j_d, face_idx_dst]
                        ny_dst = Array(dst_b.nyk)[i_d, j_d, face_idx_dst]
                        nz_dst = Array(dst_b.nzk)[i_d, j_d, face_idx_dst]

                        i_s = i_d
                        j_s = j_d

                        Area_src = Array(src_b.Areak)[i_s, j_s, face_idx_src]
                        nx_src = Array(src_b.nxk)[i_s, j_s, face_idx_src]
                        ny_src = Array(src_b.nyk)[i_s, j_s, face_idx_src]
                        nz_src = Array(src_b.nzk)[i_s, j_s, face_idx_src]

                        Area_err = abs(Area_dst - Area_src) / max(Area_dst, 1.0e-30)
                        nx_err = abs(nx_dst + nx_src)
                        ny_err = abs(ny_dst + ny_src)
                        nz_err = abs(nz_dst + nz_src)

                        if i_off == 0 && j_off == 0
                            println(io, "# i_d=$i_d, j_d=$j_d (first sample)")
                            println(io, "#   Area_dst=$Area_dst, Area_src=$Area_src, rel_err=$Area_err")
                            println(io, "#   n_dst=($nx_dst, $ny_dst, $nz_dst)")
                            println(io, "#   n_src=($nx_src, $ny_src, $nz_src)")
                            println(io, "#   normal_opposite_err=($nx_err, $ny_err, $nz_err)")
                        end

                        if Area_err > 1.0e-6 || nx_err > 1.0e-6 || ny_err > 1.0e-6 || nz_err > 1.0e-6
                            println(io, "# WARNING: Large mismatch at i=$i_d, j=$j_d")
                            println(io, "#   Area_err=$Area_err, normal_err=($nx_err, $ny_err, $nz_err)")
                        end
                    end
                end
            end

            println(io, "#")
            println(io, "# DONE")
        end
        println("  [Rank $world_rank] Metric check: $fname")
    end
end

"""
    check_gcl(blocks, world_rank)

Check Geometric Conservation Law: for each interior cell,
Σ (n_x * Area) over all faces should be zero (divergence-free metric identity).
"""
function check_gcl(blocks, world_rank)
    mkpath("debug")

    for (bid, b) in blocks
        nxp, nyp, nzp = b.Nx, b.Ny, b.Nz
        max_err = 0.0
        max_i, max_j, max_k = 0, 0, 0

        # Check interior cells only (skip ghost)
        for k in NG+1:nzp+NG
            for j in NG+1:nyp+NG
                for i in NG+1:nxp+NG
                    # Sum of (nx * Area) over 6 faces
                    # i-faces: i and i+1
                    sum_nxA = (
                        Array(b.nxi)[i+1, j, k] * Array(b.Areai)[i+1, j, k] -
                        Array(b.nxi)[i, j, k] * Array(b.Areai)[i, j, k] +
                        Array(b.nxj)[i, j+1, k] * Array(b.Areaj)[i, j+1, k] -
                        Array(b.nxj)[i, j, k] * Array(b.Areaj)[i, j, k] +
                        Array(b.nxk)[i, j, k+1] * Array(b.Areak)[i, j, k+1] -
                        Array(b.nxk)[i, j, k] * Array(b.Areak)[i, j, k]
                    )
                    # Similarly for ny*Area and nz*Area
                    sum_nyA = (
                        Array(b.nyi)[i+1, j, k] * Array(b.Areai)[i+1, j, k] -
                        Array(b.nyi)[i, j, k] * Array(b.Areai)[i, j, k] +
                        Array(b.nyj)[i, j+1, k] * Array(b.Areaj)[i, j+1, k] -
                        Array(b.nyj)[i, j, k] * Array(b.Areaj)[i, j, k] +
                        Array(b.nyk)[i, j, k+1] * Array(b.Areak)[i, j, k+1] -
                        Array(b.nyk)[i, j, k] * Array(b.Areak)[i, j, k]
                    )
                    sum_nzA = (
                        Array(b.nzi)[i+1, j, k] * Array(b.Areai)[i+1, j, k] -
                        Array(b.nzi)[i, j, k] * Array(b.Areai)[i, j, k] +
                        Array(b.nzj)[i, j+1, k] * Array(b.Areaj)[i, j+1, k] -
                        Array(b.nzj)[i, j, k] * Array(b.Areaj)[i, j, k] +
                        Array(b.nzk)[i, j, k+1] * Array(b.Areak)[i, j, k+1] -
                        Array(b.nzk)[i, j, k] * Array(b.Areak)[i, j, k]
                    )

                    err = sqrt(sum_nxA^2 + sum_nyA^2 + sum_nzA^2)
                    if err > max_err
                        max_err = err
                        max_i, max_j, max_k = i, j, k
                    end
                end
            end
        end

        fname = "debug/gcl_b$(bid)_rank$(world_rank).txt"
        open(fname, "w") do io
            println(io, "# GCL check for Block $bid, Rank $world_rank")
            println(io, "# Max |Σ n·Area| = $max_err at cell ($max_i, $max_j, $max_k)")
            println(io, "# Grid: $(nxp) x $(nyp) x $(nzp)")
            if max_err > 1.0e-10
                println(io, "# WARNING: GCL violation detected!")
            else
                println(io, "# GCL satisfied (error < 1e-10)")
            end
        end
        println("  [Rank $world_rank] GCL check block $bid: max_err=$max_err at ($max_i, $max_j, $max_k)")
    end
end

println("[check_metrics.jl] Diagnostic functions loaded.")
println("  - check_metric_consistency(blocks, connectivity, world_rank)")
println("  - check_gcl(blocks, world_rank)")
