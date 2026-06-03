# =============================================================================
# dump_ghost_interface.jl — Phase 1 Diagnostic: Ghost Cell Interface Check
#
# Dumps Q values at block interfaces to verify ghost cell exchange integrity.
# Run after sync_blocks! to check if ghost cells match neighbor's interior.
#
# Usage: include this file after sync_blocks! is defined, then call
#        dump_ghost_interface(blocks, connectivity, world_rank, tt)
# =============================================================================

"""
    dump_ghost_interface(blocks, connectivity, world_rank, tt)

Dump Q values at block interfaces to verify ghost cell exchange.
For each inter-block face, compares:
  - Block A's last interior cell
  - Block B's ghost cell (which should equal A's interior)
"""
function dump_ghost_interface(blocks, connectivity, world_rank, tt)
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

        # Sample at mid-plane of the face
        if dst_fid == 1 || dst_fid == 2  # ξ interface
            j_mid = div(nyp_d, 2) + NG
            k_mid = div(nzp_d, 2) + NG

            if dst_fid == 1  # ξ-lo: ghost cells at i = 1..NG
                # Source should be ξ-hi of src: interior at i = nxp_s+NG
                i_ghost = NG  # last ghost cell
                i_src_int = nxp_s + NG  # last interior of source
                i_dst_int = NG + 1  # first interior of dest

                fname = "debug/ghost_intf_t$(tt)_rank$(world_rank)_b$(src_b_id)_f$(src_fid)_to_b$(dst_b_id)_f$(dst_fid).txt"
                open(fname, "w") do io
                    println(io, "# Ghost interface check: Block $(src_b_id) face $(src_fid) → Block $(dst_b_id) face $(dst_fid)")
                    println(io, "# Time step: $tt, Rank: $world_rank")
                    println(io, "# Sample at j=$j_mid, k=$k_mid")
                    println(io, "#")
                    println(io, "# Source interior (src_b[$i_src_int, $j_mid, $k_mid, :])")
                    println(io, "# Dest ghost    (dst_b[$i_ghost, $j_mid, $k_mid, :])")
                    println(io, "# Dest interior (dst_b[$i_dst_int, $j_mid, $k_mid, :])")
                    println(io, "#")
                    println(io, "# var   src_interior    dst_ghost       dst_interior    ghost_error")
                    for n in 1:Nprim
                        src_val = Array(src_b.Q)[i_src_int, j_mid, k_mid, n]
                        dst_ghost = Array(dst_b.Q)[i_ghost, j_mid, k_mid, n]
                        dst_val = Array(dst_b.Q)[i_dst_int, j_mid, k_mid, n]
                        err = abs(dst_ghost - src_val)
                        @printf(io, "  %d   %14.6e %14.6e %14.6e %14.6e\n", n, src_val, dst_ghost, dst_val, err)
                    end
                end
                println("  [Rank $world_rank] Ghost interface dump: $fname")

            elseif dst_fid == 2  # ξ-hi: ghost cells at i = nxp_d+NG+1..nxp_d+2NG
                # Source should be ξ-lo of src: interior at i = NG+1
                i_ghost = nxp_d + NG + 1  # first ghost cell
                i_src_int = NG + 1  # first interior of source
                i_dst_int = nxp_d + NG  # last interior of dest

                fname = "debug/ghost_intf_t$(tt)_rank$(world_rank)_b$(src_b_id)_f$(src_fid)_to_b$(dst_b_id)_f$(dst_fid).txt"
                open(fname, "w") do io
                    println(io, "# Ghost interface check: Block $(src_b_id) face $(src_fid) → Block $(dst_b_id) face $(dst_fid)")
                    println(io, "# Time step: $tt, Rank: $world_rank")
                    println(io, "# Sample at j=$j_mid, k=$k_mid")
                    println(io, "#")
                    println(io, "# Source interior (src_b[$i_src_int, $j_mid, $k_mid, :])")
                    println(io, "# Dest ghost    (dst_b[$i_ghost, $j_mid, $k_mid, :])")
                    println(io, "# Dest interior (dst_b[$i_dst_int, $j_mid, $k_mid, :])")
                    println(io, "#")
                    println(io, "# var   src_interior    dst_ghost       dst_interior    ghost_error")
                    for n in 1:Nprim
                        src_val = Array(src_b.Q)[i_src_int, j_mid, k_mid, n]
                        dst_ghost = Array(dst_b.Q)[i_ghost, j_mid, k_mid, n]
                        dst_val = Array(dst_b.Q)[i_dst_int, j_mid, k_mid, n]
                        err = abs(dst_ghost - src_val)
                        @printf(io, "  %d   %14.6e %14.6e %14.6e %14.6e\n", n, src_val, dst_ghost, dst_val, err)
                    end
                end
                println("  [Rank $world_rank] Ghost interface dump: $fname")
            end

        elseif dst_fid == 3 || dst_fid == 4  # η interface
            i_mid = div(nxp_d, 2) + NG
            k_mid = div(nzp_d, 2) + NG

            if dst_fid == 3  # η-lo
                j_ghost = NG
                j_src_int = nyp_s + NG
                j_dst_int = NG + 1

                fname = "debug/ghost_intf_t$(tt)_rank$(world_rank)_b$(src_b_id)_f$(src_fid)_to_b$(dst_b_id)_f$(dst_fid).txt"
                open(fname, "w") do io
                    println(io, "# Ghost interface check: Block $(src_b_id) face $(src_fid) → Block $(dst_b_id) face $(dst_fid)")
                    println(io, "# Time step: $tt, Rank: $world_rank")
                    println(io, "# Sample at i=$i_mid, k=$k_mid")
                    println(io, "#")
                    println(io, "# var   src_interior    dst_ghost       dst_interior    ghost_error")
                    for n in 1:Nprim
                        src_val = Array(src_b.Q)[i_mid, j_src_int, k_mid, n]
                        dst_ghost = Array(dst_b.Q)[i_mid, j_ghost, k_mid, n]
                        dst_val = Array(dst_b.Q)[i_mid, j_dst_int, k_mid, n]
                        err = abs(dst_ghost - src_val)
                        @printf(io, "  %d   %14.6e %14.6e %14.6e %14.6e\n", n, src_val, dst_ghost, dst_val, err)
                    end
                end
                println("  [Rank $world_rank] Ghost interface dump: $fname")

            elseif dst_fid == 4  # η-hi
                j_ghost = nyp_d + NG + 1
                j_src_int = NG + 1
                j_dst_int = nyp_d + NG

                fname = "debug/ghost_intf_t$(tt)_rank$(world_rank)_b$(src_b_id)_f$(src_fid)_to_b$(dst_b_id)_f$(dst_fid).txt"
                open(fname, "w") do io
                    println(io, "# Ghost interface check: Block $(src_b_id) face $(src_fid) → Block $(dst_b_id) face $(dst_fid)")
                    println(io, "# Time step: $tt, Rank: $world_rank")
                    println(io, "# Sample at i=$i_mid, k=$k_mid")
                    println(io, "#")
                    println(io, "# var   src_interior    dst_ghost       dst_interior    ghost_error")
                    for n in 1:Nprim
                        src_val = Array(src_b.Q)[i_mid, j_src_int, k_mid, n]
                        dst_ghost = Array(dst_b.Q)[i_mid, j_ghost, k_mid, n]
                        dst_val = Array(dst_b.Q)[i_mid, j_dst_int, k_mid, n]
                        err = abs(dst_ghost - src_val)
                        @printf(io, "  %d   %14.6e %14.6e %14.6e %14.6e\n", n, src_val, dst_ghost, dst_val, err)
                    end
                end
                println("  [Rank $world_rank] Ghost interface dump: $fname")
            end

        elseif dst_fid == 5 || dst_fid == 6  # ζ interface
            i_mid = div(nxp_d, 2) + NG
            j_mid = div(nyp_d, 2) + NG

            if dst_fid == 5  # ζ-lo
                k_ghost = NG
                k_src_int = nzp_s + NG
                k_dst_int = NG + 1

                fname = "debug/ghost_intf_t$(tt)_rank$(world_rank)_b$(src_b_id)_f$(src_fid)_to_b$(dst_b_id)_f$(dst_fid).txt"
                open(fname, "w") do io
                    println(io, "# Ghost interface check: Block $(src_b_id) face $(src_fid) → Block $(dst_b_id) face $(dst_fid)")
                    println(io, "# Time step: $tt, Rank: $world_rank")
                    println(io, "# Sample at i=$i_mid, j=$j_mid")
                    println(io, "#")
                    println(io, "# var   src_interior    dst_ghost       dst_interior    ghost_error")
                    for n in 1:Nprim
                        src_val = Array(src_b.Q)[i_mid, j_mid, k_src_int, n]
                        dst_ghost = Array(dst_b.Q)[i_mid, j_mid, k_ghost, n]
                        dst_val = Array(dst_b.Q)[i_mid, j_mid, k_dst_int, n]
                        err = abs(dst_ghost - src_val)
                        @printf(io, "  %d   %14.6e %14.6e %14.6e %14.6e\n", n, src_val, dst_ghost, dst_val, err)
                    end
                end
                println("  [Rank $world_rank] Ghost interface dump: $fname")

            elseif dst_fid == 6  # ζ-hi
                k_ghost = nzp_d + NG + 1
                k_src_int = NG + 1
                k_dst_int = nzp_d + NG

                fname = "debug/ghost_intf_t$(tt)_rank$(world_rank)_b$(src_b_id)_f$(src_fid)_to_b$(dst_b_id)_f$(dst_fid).txt"
                open(fname, "w") do io
                    println(io, "# Ghost interface check: Block $(src_b_id) face $(src_fid) → Block $(dst_b_id) face $(dst_fid)")
                    println(io, "# Time step: $tt, Rank: $world_rank")
                    println(io, "# Sample at i=$i_mid, j=$j_mid")
                    println(io, "#")
                    println(io, "# var   src_interior    dst_ghost       dst_interior    ghost_error")
                    for n in 1:Nprim
                        src_val = Array(src_b.Q)[i_mid, j_mid, k_src_int, n]
                        dst_ghost = Array(dst_b.Q)[i_mid, j_mid, k_ghost, n]
                        dst_val = Array(dst_b.Q)[i_mid, j_mid, k_dst_int, n]
                        err = abs(dst_ghost - src_val)
                        @printf(io, "  %d   %14.6e %14.6e %14.6e %14.6e\n", n, src_val, dst_ghost, dst_val, err)
                    end
                end
                println("  [Rank $world_rank] Ghost interface dump: $fname")
            end
        end
    end
end

"""
    dump_interface_flux(blocks, connectivity, shared_Fx, shared_Fy, shared_Fz, world_rank, tt)

Dump flux values at inter-block faces to verify flux conservation.
For conservation: F_A(face) + F_B(face) should be zero (opposite normals).
"""
function dump_interface_flux(blocks, connectivity, shared_Fx, shared_Fy, shared_Fz, world_rank, tt)
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

        # Sample at mid-plane
        if dst_fid == 1 || dst_fid == 2  # ξ interface
            j_mid = div(nyp_d, 2) + NG
            k_mid = div(nzp_d, 2) + NG

            if dst_fid == 1
                i_face_dst = NG  # face between ghost and first interior
                i_face_src = nxp_s + NG  # face at end of source
            else
                i_face_dst = nxp_d + NG  # face at end of dest
                i_face_src = NG  # face at start of source
            end

            fname = "debug/flux_intf_t$(tt)_rank$(world_rank)_b$(src_b_id)_f$(src_fid)_to_b$(dst_b_id)_f$(dst_fid).txt"
            open(fname, "w") do io
                println(io, "# Flux interface check: Block $(src_b_id) face $(src_fid) → Block $(dst_b_id) face $(dst_fid)")
                println(io, "# Time step: $tt, Rank: $world_rank")
                println(io, "# Sample at j=$j_mid, k=$k_mid")
                println(io, "#")
                println(io, "# var   src_flux         dst_flux         sum              relative_error")
                for n in 1:Ncons
                    src_F = Array(shared_Fx)[i_face_src, j_mid, k_mid, n]
                    dst_F = Array(shared_Fx)[i_face_dst, j_mid, k_mid, n]
                    sum_F = src_F + dst_F
                    ref = max(abs(src_F), abs(dst_F), 1.0e-30)
                    rel_err = abs(sum_F) / ref
                    @printf(io, "  %d   %14.6e %14.6e %14.6e %14.6e\n", n, src_F, dst_F, sum_F, rel_err)
                end
            end
            println("  [Rank $world_rank] Flux interface dump: $fname")

        elseif dst_fid == 3 || dst_fid == 4  # η interface
            i_mid = div(nxp_d, 2) + NG
            k_mid = div(nzp_d, 2) + NG

            if dst_fid == 3
                j_face_dst = NG
                j_face_src = nyp_s + NG
            else
                j_face_dst = nyp_d + NG
                j_face_src = NG
            end

            fname = "debug/flux_intf_t$(tt)_rank$(world_rank)_b$(src_b_id)_f$(src_fid)_to_b$(dst_b_id)_f$(dst_fid).txt"
            open(fname, "w") do io
                println(io, "# Flux interface check: Block $(src_b_id) face $(src_fid) → Block $(dst_b_id) face $(dst_fid)")
                println(io, "# Time step: $tt, Rank: $world_rank")
                println(io, "# Sample at i=$i_mid, k=$k_mid")
                println(io, "#")
                println(io, "# var   src_flux         dst_flux         sum              relative_error")
                for n in 1:Ncons
                    src_F = Array(shared_Fy)[i_mid, j_face_src, k_mid, n]
                    dst_F = Array(shared_Fy)[i_mid, j_face_dst, k_mid, n]
                    sum_F = src_F + dst_F
                    ref = max(abs(src_F), abs(dst_F), 1.0e-30)
                    rel_err = abs(sum_F) / ref
                    @printf(io, "  %d   %14.6e %14.6e %14.6e %14.6e\n", n, src_F, dst_F, sum_F, rel_err)
                end
            end
            println("  [Rank $world_rank] Flux interface dump: $fname")

        elseif dst_fid == 5 || dst_fid == 6  # ζ interface
            i_mid = div(nxp_d, 2) + NG
            j_mid = div(nyp_d, 2) + NG

            if dst_fid == 5
                k_face_dst = NG
                k_face_src = nzp_s + NG
            else
                k_face_dst = nzp_d + NG
                k_face_src = NG
            end

            fname = "debug/flux_intf_t$(tt)_rank$(world_rank)_b$(src_b_id)_f$(src_fid)_to_b$(dst_b_id)_f$(dst_fid).txt"
            open(fname, "w") do io
                println(io, "# Flux interface check: Block $(src_b_id) face $(src_fid) → Block $(dst_b_id) face $(dst_fid)")
                println(io, "# Time step: $tt, Rank: $world_rank")
                println(io, "# Sample at i=$i_mid, j=$j_mid")
                println(io, "#")
                println(io, "# var   src_flux         dst_flux         sum              relative_error")
                for n in 1:Ncons
                    src_F = Array(shared_Fz)[i_mid, j_mid, k_face_src, n]
                    dst_F = Array(shared_Fz)[i_mid, j_mid, k_face_dst, n]
                    sum_F = src_F + dst_F
                    ref = max(abs(src_F), abs(dst_F), 1.0e-30)
                    rel_err = abs(sum_F) / ref
                    @printf(io, "  %d   %14.6e %14.6e %14.6e %14.6e\n", n, src_F, dst_F, sum_F, rel_err)
                end
            end
            println("  [Rank $world_rank] Flux interface dump: $fname")
        end
    end
end

println("[dump_ghost_interface.jl] Diagnostic functions loaded.")
println("  - dump_ghost_interface(blocks, connectivity, world_rank, tt)")
println("  - dump_interface_flux(blocks, connectivity, shared_Fx, shared_Fy, shared_Fz, world_rank, tt)")
