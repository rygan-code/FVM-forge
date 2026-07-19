@inline function mhd_reflect_wall_field(
    bx, by, bz, nx, ny, nz, preserve_normal::Bool,
)
    bn = bx*nx + by*ny + bz*nz
    if preserve_normal
        return 2bn*nx - bx, 2bn*ny - by, 2bn*nz - bz
    end
    return bx - 2bn*nx, by - 2bn*ny, bz - 2bn*nz
end
