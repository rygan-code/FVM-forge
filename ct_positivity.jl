# Observation-only positivity diagnostics for the strict CT+HLLD path.

const CT_POS_META_LEN = 10
const CT_POS_VALUE_LEN = 17
const CT_POS_SITE_RECONSTRUCTED = Int32(1)
const CT_POS_SITE_FACE_B_INVARIANT = Int32(2)
const CT_POS_SITE_HLLD_INPUT = Int32(3)
const CT_POS_SITE_POST_HYDRO = Int32(4)
const CT_POS_SITE_POST_CT_SYNC = Int32(5)
const CT_POS_SITE_GHOST_REFRESH = Int32(6)
const CT_POS_SITE_PRE_CT_SYNC = Int32(7)
const CT_POS_WENO_TO_PLM_COUNT = 8
const CT_POS_PLM_TO_FIRST_COUNT = 9
const CT_POS_HLLD_TO_HLLE_COUNT = 10

@inline function _ct_try_claim!(meta)
    @static if @isdefined(gpu_atomic_cas!)
        return gpu_atomic_cas!(
            meta, 1, Int32(0), Int32(1),
        ) == Int32(0)
    elseif @isdefined(USE_CUDA) && USE_CUDA
        return CUDA.atomic_cas!(pointer(meta), Int32(0), Int32(1)) == Int32(0)
    else
        return Core.Intrinsics.atomic_pointerreplace(
            pointer(meta), Int32(0), Int32(1),
            :acquire_release, :acquire,
        ).success
    end
end

@inline function ct_record_fallback!(meta, counter_index::Integer)
    @static if @isdefined(gpu_atomic_add!)
        gpu_atomic_add!(meta, counter_index, Int32(1))
    else
        pointer_value = pointer(meta, counter_index)
        old = @inbounds meta[counter_index]
        while true
            result = Core.Intrinsics.atomic_pointerreplace(
                pointer_value, old, old + Int32(1),
                :acquire_release, :acquire,
            )
            result.success && break
            old = result.old
        end
    end
    return nothing
end

@inline function _ct_store_failure!(
    meta, values,
    site::Int32, direction::Int32, side::Int32,
    i::Int32, j::Int32, k::Int32,
    U, bface, rho, kinetic, magnetic, internal, pressure,
)
    if _ct_try_claim!(meta)
        @inbounds begin
            meta[2] = site
            meta[3] = direction
            meta[4] = side
            meta[5] = i
            meta[6] = j
            meta[7] = k

            values[1] = rho
            values[2] = kinetic
            values[3] = magnetic
            values[4] = internal
            values[5] = pressure
            values[6] = U[6]
            values[7] = U[7]
            values[8] = U[8]
            values[9] = bface
            values[10] = U[5]
        end
    end
    return nothing
end

@inline function ct_record_state_if_invalid!(
    meta, values,
    site::Int32, direction::Int32, side::Int32,
    i::Int32, j::Int32, k::Int32,
    U, bface, gamma,
)
    rho, kinetic, magnetic, internal, pressure = mhd_raw_thermo(U, gamma)
    valid = isfinite(rho) && rho > zero(rho) &&
            isfinite(internal) && internal > zero(internal) &&
            isfinite(pressure) && pressure > zero(pressure)
    if valid
        return false
    end

    _ct_store_failure!(
        meta, values, site, direction, side, i, j, k, U, bface,
        rho, kinetic, magnetic, internal, pressure,
    )
    return true
end

@inline function _ct_record_if_invalid!(
    meta, values,
    site::Int32, direction::Int32, side::Int32,
    i::Int32, j::Int32, k::Int32,
    U, bface,
)
    return ct_record_state_if_invalid!(
        meta, values, site, direction, side, i, j, k, U, bface, γ,
    )
end

@inline function ct_record_cell_values_if_invalid!(
    meta, values, site::Int32, i::Int32, j::Int32, k::Int32,
    rho, momentum_x, momentum_y, momentum_z, energy,
    magnetic_x, magnetic_y, magnetic_z, psi, gamma,
)
    rho_raw, kinetic, magnetic, internal, pressure =
        mhd_raw_thermo_components(
            rho, momentum_x, momentum_y, momentum_z, energy,
            magnetic_x, magnetic_y, magnetic_z, gamma,
        )
    valid = isfinite(rho_raw) && rho_raw > zero(rho_raw) &&
            isfinite(internal) && internal > zero(internal) &&
            isfinite(pressure) && pressure > zero(pressure)
    if valid
        return false
    end

    if _ct_try_claim!(meta)
        @inbounds begin
            meta[2] = site
            meta[3] = Int32(0)
            meta[4] = Int32(0)
            meta[5] = i
            meta[6] = j
            meta[7] = k

            values[1] = rho_raw
            values[2] = kinetic
            values[3] = magnetic
            values[4] = internal
            values[5] = pressure
            values[6] = magnetic_x
            values[7] = magnetic_y
            values[8] = magnetic_z
            values[9] = oftype(rho_raw, NaN)
            values[10] = energy
        end
    end
    return true
end

function ct_check_cell_positivity_kernel!(
    meta, values, U, gamma::FT, site::Int32, include_ghost::Int32,
    nxp::Int32, nyp::Int32, nzp::Int32,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)

    if include_ghost == Int32(0)
        if i > nxp || j > nyp || k > nzp
            return
        end
        ii = i + ng
        jj = j + ng
        kk = k + ng
    else
        if i > nxp + Int32(2)*ng ||
           j > nyp + Int32(2)*ng ||
           k > nzp + Int32(2)*ng
            return
        end
        ii, jj, kk = i, j, k
    end

    @inbounds begin
        rho = U[ii, jj, kk, 1]
        momentum_x = U[ii, jj, kk, 2]
        momentum_y = U[ii, jj, kk, 3]
        momentum_z = U[ii, jj, kk, 4]
        energy = U[ii, jj, kk, 5]
        magnetic_x = U[ii, jj, kk, 6]
        magnetic_y = U[ii, jj, kk, 7]
        magnetic_z = U[ii, jj, kk, 8]
        psi = U[ii, jj, kk, 9]
    end
    ct_record_cell_values_if_invalid!(
        meta, values, site, Int32(ii), Int32(jj), Int32(kk),
        rho, momentum_x, momentum_y, momentum_z, energy,
        magnetic_x, magnetic_y, magnetic_z, psi, gamma,
    )
    return
end

function ct_check_sync_transition_kernel!(
    meta, values, U, Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    gamma::FT, nxp::Int32, nyp::Int32, nzp::Int32,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii = i + Int32(NG)
    jj = j + Int32(NG)
    kk = k + Int32(NG)
    @inbounds begin
        rho = U[ii, jj, kk, 1]
        momentum_x = U[ii, jj, kk, 2]
        momentum_y = U[ii, jj, kk, 3]
        momentum_z = U[ii, jj, kk, 4]
        energy = U[ii, jj, kk, 5]
        old_bx = U[ii, jj, kk, 6]
        old_by = U[ii, jj, kk, 7]
        old_bz = U[ii, jj, kk, 8]
        area_i_lo = SVector(
            Areai[ii,jj,kk]*nxi[ii,jj,kk],
            Areai[ii,jj,kk]*nyi[ii,jj,kk],
            Areai[ii,jj,kk]*nzi[ii,jj,kk],
        )
        area_i_hi = SVector(
            Areai[ii+Int32(1),jj,kk]*nxi[ii+Int32(1),jj,kk],
            Areai[ii+Int32(1),jj,kk]*nyi[ii+Int32(1),jj,kk],
            Areai[ii+Int32(1),jj,kk]*nzi[ii+Int32(1),jj,kk],
        )
        area_j_lo = SVector(
            Areaj[ii,jj,kk]*nxj[ii,jj,kk],
            Areaj[ii,jj,kk]*nyj[ii,jj,kk],
            Areaj[ii,jj,kk]*nzj[ii,jj,kk],
        )
        area_j_hi = SVector(
            Areaj[ii,jj+Int32(1),kk]*nxj[ii,jj+Int32(1),kk],
            Areaj[ii,jj+Int32(1),kk]*nyj[ii,jj+Int32(1),kk],
            Areaj[ii,jj+Int32(1),kk]*nzj[ii,jj+Int32(1),kk],
        )
        area_k_lo = SVector(
            Areak[ii,jj,kk]*nxk[ii,jj,kk],
            Areak[ii,jj,kk]*nyk[ii,jj,kk],
            Areak[ii,jj,kk]*nzk[ii,jj,kk],
        )
        area_k_hi = SVector(
            Areak[ii,jj,kk+Int32(1)]*nxk[ii,jj,kk+Int32(1)],
            Areak[ii,jj,kk+Int32(1)]*nyk[ii,jj,kk+Int32(1)],
            Areak[ii,jj,kk+Int32(1)]*nzk[ii,jj,kk+Int32(1)],
        )
        recovered_b = ct_recover_cell_b(
            area_i_lo, area_i_hi, area_j_lo, area_j_hi, area_k_lo, area_k_hi,
            Bx_face[ii,jj,kk], Bx_face[ii+Int32(1),jj,kk],
            By_face[ii,jj,kk], By_face[ii,jj+Int32(1),kk],
            Bz_face[ii,jj,kk], Bz_face[ii,jj,kk+Int32(1)],
        )
        new_bx = recovered_b[1]
        new_by = recovered_b[2]
        new_bz = recovered_b[3]
    end

    _, kinetic, new_magnetic, new_internal, new_pressure =
        mhd_raw_thermo_components(
            rho, momentum_x, momentum_y, momentum_z, energy,
            new_bx, new_by, new_bz, gamma,
        )
    if isfinite(rho) && rho > zero(FT) &&
       isfinite(new_internal) && new_internal > zero(FT) &&
       isfinite(new_pressure) && new_pressure > zero(FT)
        return
    end

    _, _, _, _, old_pressure = mhd_raw_thermo_components(
        rho, momentum_x, momentum_y, momentum_z, energy,
        old_bx, old_by, old_bz, gamma,
    )

    if _ct_try_claim!(meta)
        @inbounds begin
            meta[2] = CT_POS_SITE_PRE_CT_SYNC
            meta[3] = Int32(0)
            meta[4] = Int32(0)
            meta[5] = ii
            meta[6] = jj
            meta[7] = kk
            values[1] = rho
            values[2] = kinetic
            values[3] = new_magnetic
            values[4] = new_internal
            values[5] = new_pressure
            values[6] = new_bx
            values[7] = new_by
            values[8] = new_bz
            values[9] = old_pressure
            values[10] = energy
            values[11] = old_bx
            values[12] = old_by
            values[13] = old_bz
            values[14] = FT(NaN)
            values[15] = FT(NaN)
            values[16] = FT(NaN)
            values[17] = FT(NaN)
        end
    end
    return
end

@inline function _ct_record_pressure_change!(
    meta, values,
    direction::Int32, side::Int32,
    i::Int32, j::Int32, k::Int32,
    before, after, bface,
)
    _, _, _, _, p_before = mhd_raw_thermo(before, γ)
    rho, kinetic, magnetic, internal, p_after = mhd_raw_thermo(after, γ)
    tol = FT(64) * eps(FT) * max(one(FT), abs(p_before))
    changed = !(isfinite(p_before) && isfinite(p_after)) ||
              abs(p_after - p_before) > tol
    if !changed
        return false
    end

    _ct_store_failure!(
        meta, values, CT_POS_SITE_FACE_B_INVARIANT,
        direction, side, i, j, k, after, bface,
        rho, kinetic, magnetic, internal, p_after,
    )
    return true
end

function ct_reset_positivity!(meta, values)
    fill!(meta, Int32(0))
    fill!(values, zero(eltype(values)))
    return nothing
end

function ct_positivity_snapshot(meta, values)
    return Array(meta), Array(values)
end

function ct_fallback_counts(meta)
    meta_h = Array(meta)
    return (
        weno_to_plm=Int(meta_h[CT_POS_WENO_TO_PLM_COUNT]),
        plm_to_first=Int(meta_h[CT_POS_PLM_TO_FIRST_COUNT]),
        hlld_to_hlle=Int(meta_h[CT_POS_HLLD_TO_HLLE_COUNT]),
    )
end

function ct_check_positivity_or_abort!(
    meta, values;
    rank::Integer, block::Integer, step::Integer, rk_stage::Integer,
)
    meta_h, values_h = ct_positivity_snapshot(meta, values)
    if meta_h[1] == Int32(0)
        return false
    end

    site_names = (
        "reconstruction before face-B",
        "face-B pressure invariant",
        "HLLD input",
        "post-hydro/pre-CT cell",
        "post-CT-sync cell",
        "ghost primitive refresh",
        "pre-CT-sync B/energy transition",
    )
    site = Int(meta_h[2])
    site_name = 1 <= site <= length(site_names) ? site_names[site] : "unknown"
    message =
        "STRICT CT POSITIVITY VIOLATION: rank=$(rank) block=$(block) " *
        "step=$(step) rk_stage=$(rk_stage) site=$(site)($(site_name)) " *
        "direction=$(meta_h[3]) side=$(meta_h[4]) " *
        "index=($(meta_h[5]),$(meta_h[6]),$(meta_h[7]))\n" *
        "  rho=$(values_h[1]) KE=$(values_h[2]) ME=$(values_h[3]) " *
        "raw_ei=$(values_h[4]) raw_p=$(values_h[5])\n" *
        "  B=($(values_h[6]),$(values_h[7]),$(values_h[8])) " *
        "face_B=$(values_h[9]) U5=$(values_h[10])"
    if site == Int(CT_POS_SITE_PRE_CT_SYNC)
        message *=
            "\n  old_B=($(values_h[11]),$(values_h[12]),$(values_h[13])) " *
            "old_p=$(values_h[9])"
    end
    printstyled(message * "\n", color=:red)
    flush(stdout)
    MPI.Abort(MPI.COMM_WORLD, 86)
    return true
end
