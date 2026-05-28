using Pkg
Pkg.activate(".")
packages = ["MPI", "CUDA", "StaticArrays", "HDF5", "DelimitedFiles", "WriteVTK", "Dates", "Printf"]
for pkg in packages
    Pkg.add(pkg)
end
Pkg.instantiate()
Pkg.precompile()
