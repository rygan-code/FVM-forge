# Shared coordinate sampling helpers for structured node-based meshes.

@inline function structured_cell_center_coordinates(x, y, z, i, j, k)
    scale = one(eltype(x)) / 8
    @inbounds return SVector(
        scale * (
            x[i,   j,   k]   + x[i+1, j,   k] +
            x[i,   j+1, k]   + x[i+1, j+1, k] +
            x[i,   j,   k+1] + x[i+1, j,   k+1] +
            x[i,   j+1, k+1] + x[i+1, j+1, k+1]
        ),
        scale * (
            y[i,   j,   k]   + y[i+1, j,   k] +
            y[i,   j+1, k]   + y[i+1, j+1, k] +
            y[i,   j,   k+1] + y[i+1, j,   k+1] +
            y[i,   j+1, k+1] + y[i+1, j+1, k+1]
        ),
        scale * (
            z[i,   j,   k]   + z[i+1, j,   k] +
            z[i,   j+1, k]   + z[i+1, j+1, k] +
            z[i,   j,   k+1] + z[i+1, j,   k+1] +
            z[i,   j+1, k+1] + z[i+1, j+1, k+1]
        ),
    )
end
