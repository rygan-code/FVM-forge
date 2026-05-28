using HDF5, Statistics

function check_metrics(filename)
    println("Checking metrics in $filename...")
    h5open(filename, "r") do file
        for key in keys(file)
            data = read(file, key)
            if eltype(data) <: AbstractFloat
                min_val = minimum(data)
                max_val = maximum(data)
                nan_count = count(isnan, data)
                
                println("$key: min=$min_val, max=$max_val, NaNs=$nan_count")
                
                if nan_count > 0
                    println("ERROR: NaNs found in $key!")
                end
                
                if (key == "Areai" || key == "Areaj" || key == "Areak" || key == "Vol") && min_val <= 0
                    println("WARNING: Non-positive values found in $key!")
                end
            end
        end
    end
end

if isfile("MESH/oblique_metrics.h5")
    check_metrics("MESH/oblique_metrics.h5")
else
    println("MESH/oblique_metrics.h5 not found")
end
