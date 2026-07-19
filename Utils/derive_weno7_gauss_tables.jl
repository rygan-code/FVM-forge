using LinearAlgebra
using Printf

const RT = Rational{BigInt}
const CANDIDATE_STENCILS = (
    (-3, -2, -1, 0),
    (-2, -1, 0, 1),
    (-1, 0, 1, 2),
    (0, 1, 2, 3),
)
const OPTIMAL_STENCIL = (-3, -2, -1, 0, 1, 2, 3)

to_rational(value::Integer) = BigInt(value)//BigInt(1)

function inverse_vandermonde(points)
    n = length(points)
    vandermonde = Matrix{RT}(undef, n, n)
    for degree in 0:n-1, point in 1:n
        vandermonde[degree+1,point] = to_rational(points[point])^degree
    end
    inverse = inv(vandermonde)
    @assert vandermonde*inverse == Matrix{RT}(I, n, n)
    return inverse
end

function interpolation_weights(inverse_vandermonde, target::BigFloat)
    n = size(inverse_vandermonde, 1)
    powers = BigFloat[target^degree for degree in 0:n-1]
    return BigFloat.(inverse_vandermonde)*powers
end

function interpolation_weights(inverse_vandermonde, target::RT)
    n = size(inverse_vandermonde, 1)
    powers = RT[target^degree for degree in 0:n-1]
    return inverse_vandermonde*powers
end

function candidate_table(target, candidate_inverses)
    table = Matrix{BigFloat}(undef, 4, 4)
    for stencil in 1:4
        table[stencil,:] .= interpolation_weights(
            candidate_inverses[stencil], target,
        )
    end
    return table
end

function embedded_candidates(table)
    embedded = zeros(eltype(table), 7, 4)
    for stencil in 1:4
        embedded[stencil:stencil+3,stencil] .= table[stencil,:]
    end
    return embedded
end

function point_tables(candidate_inverses, optimal_inverse)
    target = BigInt(1)//BigInt(2)
    candidates = Matrix{RT}(undef, 4, 4)
    for stencil in 1:4
        candidates[stencil,:] .= interpolation_weights(
            candidate_inverses[stencil], target,
        )
    end
    optimal = interpolation_weights(optimal_inverse, target)
    embedded = embedded_candidates(candidates)
    ideal = embedded[1:4,:] \ optimal[1:4]

    expected_candidates = RT[
        -5//16 21//16 -35//16 35//16
         1//16 -5//16  15//16  5//16
        -1//16  9//16   9//16 -1//16
         5//16 15//16  -5//16  1//16
    ]
    expected_optimal = RT[-5, 42, -175, 700, 525, -70, 7] .// 1024
    expected_ideal = RT[1, 21, 35, 7] .// 64
    @assert candidates == expected_candidates
    @assert optimal == expected_optimal
    @assert ideal == expected_ideal
    @assert embedded*ideal == optimal
    @assert sum(ideal) == one(RT)
    @assert all(ideal .> zero(RT))
    return candidates, optimal, ideal
end

function ideal_weights(table, optimal)
    embedded = embedded_candidates(table)
    ideal = embedded[1:4,:] \ optimal[1:4]
    @assert maximum(abs.(embedded*ideal - optimal)) < big"1e-60"
    @assert abs(sum(ideal) - one(BigFloat)) < big"1e-60"
    return ideal
end

function split_ideal_weights(ideal)
    if all(ideal .>= 0)
        return (
            one(BigFloat), zero(BigFloat),
            copy(ideal), zeros(BigFloat, 4),
        )
    end

    theta = BigFloat(3)
    plus_raw = (ideal + theta*abs.(ideal))/2
    minus_raw = (-ideal + theta*abs.(ideal))/2
    sigma_plus = sum(plus_raw)
    sigma_minus = sum(minus_raw)
    plus = plus_raw/sigma_plus
    minus = minus_raw/sigma_minus
    @assert maximum(abs.(sigma_plus*plus - sigma_minus*minus - ideal)) <
            big"1e-60"
    return sigma_plus, sigma_minus, plus, minus
end

float_literal(value) = @sprintf("%.17g", Float64(value))

function print_svector(name, values)
    println("const $name = SVector{$(length(values)),Float64}(")
    for value in values
        println("    $(float_literal(value)),")
    end
    println(")")
end

function print_smatrix(name, matrix)
    rows, columns = size(matrix)
    println("const $name = SMatrix{$rows,$columns,Float64}(")
    for column in 1:columns
        values = join(float_literal.(matrix[:,column]), ", ")
        println("    $values,")
    end
    println(")")
end

function print_candidate_tables(tables)
    println("const WENO7_GL4_CANDIDATE_COEFFS = (")
    for table in tables
        println("    SMatrix{4,4,Float64}(")
        for column in 1:4
            values = join(float_literal.(table[:,column]), ", ")
            println("        $values,")
        end
        println("    ),")
    end
    println(")")
end

function main()
    setprecision(BigFloat, 256) do
        root = sqrt(BigFloat(6)/5)
        outer = sqrt((BigFloat(3) + 2root)/7)/2
        inner = sqrt((BigFloat(3) - 2root)/7)/2
        nodes = BigFloat[-outer, -inner, inner, outer]
        root30 = sqrt(BigFloat(30))
        weights = BigFloat[
            (18-root30)/72, (18+root30)/72,
            (18+root30)/72, (18-root30)/72,
        ]

        candidate_inverses = inverse_vandermonde.(CANDIDATE_STENCILS)
        optimal_inverse = inverse_vandermonde(OPTIMAL_STENCIL)
        candidate_tables = candidate_table.(nodes, Ref(candidate_inverses))
        optimal_tables = interpolation_weights.(Ref(optimal_inverse), nodes)
        ideal_tables = ideal_weights.(candidate_tables, optimal_tables)
        splits = split_ideal_weights.(ideal_tables)
        point_candidates, point_optimal, point_ideal = point_tables(
            candidate_inverses, optimal_inverse,
        )

        candidate_symmetry = maximum(
            maximum(abs.(candidate_tables[q] -
                         reverse(candidate_tables[5-q], dims=(1, 2))))
            for q in 1:4
        )
        optimal_symmetry = maximum(
            maximum(abs.(optimal_tables[q] - reverse(optimal_tables[5-q])))
            for q in 1:4
        )
        ideal_symmetry = maximum(
            maximum(abs.(ideal_tables[q] - reverse(ideal_tables[5-q])))
            for q in 1:4
        )
        @assert candidate_symmetry < big"1e-70"
        @assert optimal_symmetry < big"1e-70"
        @assert ideal_symmetry < big"1e-70"

        optimal_matrix = reduce(hcat, optimal_tables)
        ideal_matrix = reduce(hcat, ideal_tables)
        sigma_plus = first.(splits)
        sigma_minus = getindex.(splits, 2)
        ideal_plus = reduce(hcat, getindex.(splits, 3))
        ideal_minus = reduce(hcat, getindex.(splits, 4))

        println("# === BEGIN GENERATED WENO7 GAUSS TABLES ===")
        print_svector("WENO7_GL4_X", nodes)
        print_svector("WENO7_GL4_W", weights)
        print_candidate_tables(candidate_tables)
        print_smatrix("WENO7_GL4_OPTIMAL_COEFFS", optimal_matrix)
        print_smatrix("WENO7_GL4_IDEAL_WEIGHTS", ideal_matrix)
        print_svector("WENO7_GL4_SIGMA_PLUS", sigma_plus)
        print_svector("WENO7_GL4_SIGMA_MINUS", sigma_minus)
        print_smatrix("WENO7_GL4_IDEAL_PLUS", ideal_plus)
        print_smatrix("WENO7_GL4_IDEAL_MINUS", ideal_minus)
        println("# === END GENERATED WENO7 GAUSS TABLES ===")
        println("# === BEGIN GENERATED WENO7 POINT TABLES ===")
        print_smatrix("WENO7_POINT_FACE_CANDIDATE_COEFFS", point_candidates)
        print_svector("WENO7_POINT_FACE_OPTIMAL_COEFFS", point_optimal)
        print_svector("WENO7_POINT_FACE_IDEAL_WEIGHTS", point_ideal)
        println("# === END GENERATED WENO7 POINT TABLES ===")
        println("# verified Rational{BigInt} Vandermonde inverses, point tables, and theta=3 splits")
    end
end

main()
