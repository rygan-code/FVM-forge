# Verify the 8^3/16^3/32^3 warped-mesh Alfven convergence suite.

using Printf

include(joinpath(@__DIR__, "metric_ct_diagnostics.jl"))

function _metric_ct_convergence_main(args)
    length(args) == 3 || error(
        "usage: julia verify_convergence.jl <N8.dat> <N16.dat> <N32.dat>",
    )
    summary = metric_ct_verify_convergence(args; min_order=1.7)

    println("metric CT Alfven convergence passed")
    for index in eachindex(summary.resolutions)
        @printf(
            "N=%d L1=%.16e L2=%.16e\n",
            summary.resolutions[index],
            summary.l1_errors[index],
            summary.l2_errors[index],
        )
    end
    @printf(
        "L1 orders: %.8f %.8f; L2 orders: %.8f %.8f\n",
        summary.l1_orders...,
        summary.l2_orders...,
    )
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    _metric_ct_convergence_main(ARGS)
end
