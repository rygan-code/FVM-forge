include(joinpath(@__DIR__,"..","src","mesh","openfoam_mesh_generator.jl"))
if abspath(PROGRAM_FILE)==@__FILE__
    output_dir=length(ARGS)>=1 ? ARGS[1] : "test_openfoam_case"
    cell_count=length(ARGS)>=2 ? parse(Int,ARGS[2]) : 20
    cell_count>0 || throw(ArgumentError("ncells must be positive"))
    generate_openfoam_mesh(output_dir,cell_count)
end
