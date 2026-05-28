using HDF5

println("=== OLD MESH B0 ===")
h5open("MESH_COARSE_OLD/mesh_b0.h5", "r") do f
    println("  Keys: ", keys(f))
    println("  Nx=", read(f["Nx"]), " Ny=", read(f["Ny"]), " Nz=", read(f["Nz"]))
end

println("=== OLD MESH B1 ===")
h5open("MESH_COARSE_OLD/mesh_b1.h5", "r") do f
    println("  Nx=", read(f["Nx"]), " Ny=", read(f["Ny"]), " Nz=", read(f["Nz"]))
end

println("\n=== NEW MESH B0 ===")
h5open("MESH_COARSE/mesh_b0.h5", "r") do f
    println("  Nx=", read(f["Nx"]), " Ny=", read(f["Ny"]), " Nz=", read(f["Nz"]))
end

println("=== NEW MESH B1 ===")
h5open("MESH_COARSE/mesh_b1.h5", "r") do f
    println("  Nx=", read(f["Nx"]), " Ny=", read(f["Ny"]), " Nz=", read(f["Nz"]))
end

println("\n=== CHK B1 structure ===")
h5open("CHK_OLD/chk-547000-b1.h5", "r") do f
    println("  Keys: ", keys(f))
    for k in keys(f)
        d = f[k]
        if isa(d, HDF5.Dataset)
            println("  ", k, ": size=", size(d), " type=", eltype(d))
        end
    end
end

println("\n=== CHK B0 structure ===")
h5open("CHK_OLD/chk-547000-b0.h5", "r") do f
    println("  Keys: ", keys(f))
    for k in keys(f)
        d = f[k]
        if isa(d, HDF5.Dataset)
            println("  ", k, ": size=", size(d), " type=", eltype(d))
        end
    end
end
