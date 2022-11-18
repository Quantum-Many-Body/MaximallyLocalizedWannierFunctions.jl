module MaximallyLocalizedWannierFunctions

using Base.Iterators: product
using LinearAlgebra: I, Diagonal, diag, dot, eigen, norm, svd
using Optim: FirstOrderOptimizer, GradientDescent
using QuantumLattices: atol, rtol, BrillouinZone, Lattice, Neighbors, bonds, expand, matrix, minimumlengths, rcoordinate
using TightBindingApproximation: TBA

import TightBindingApproximation: optimize!
import QuantumLattices: dimension, dtype, periods, update!

export δ, Hamiltonian, MLWF, rₙ, r²ₙ, Ω, Ω₁, Ω₂, optimize!

"""
    finitedifferences(vectors::AbstractVector{<:AbstractVector}; nneighbor::Int=12, coordination::Int=12) -> Tuple{Vector, Vector{Vector}}

Get the weights and difference vectors of a Monkhorst-Pack discretization scheme whose minimum mesh translations are given by `vectors`.

See Sec. 3.2 in [Reference](https://doi.org/10.1016/j.cpc.2007.11.016) for the details of the method.
"""
function finitedifferences(vectors::AbstractVector{<:AbstractVector}; nneighbor::Int=12, coordination::Int=12)
    @assert length(vectors)>0 "finitedifferences error: no input minimum Monkhorst-Pack mesh translation vectors."
    dimension, datatype = length(vectors[1]), eltype(eltype(vectors))
    candidates = bonds(
        Lattice(zeros(datatype, dimension); vectors=vectors),
        Neighbors(minimumlengths(zeros(datatype, dimension, 1), vectors, nneighbor; coordination=coordination)[2:end], 1:nneighbor)
    )
    counts, differences = zeros(Int, nneighbor), eltype(vectors)[]
    A = zeros(datatype, dimension, dimension, nneighbor)
    q = reshape(Matrix{datatype}(I, dimension, dimension), dimension*dimension)
    order = 1
    for shell = 1:nneighbor
        shell==order || (A[:, :, order] .= 0)
        num, container = 0, eltype(vectors)[]
        for candidate in candidates
            if candidate.kind==shell
                num += 2
                coordinate = rcoordinate(candidate)
                push!(container, +coordinate, -coordinate)
                for m=1:dimension, n=1:dimension
                    A[m, n, order] += 2*coordinate[m]*coordinate[n]
                end
            end
        end
        M = reshape(A[:, :, 1:order], (dimension*dimension, order))
        factorization = svd(M)
        if !isapprox(factorization.S[end], 0; atol=atol, rtol=rtol)
            counts[order] = num
            append!(differences, container)
            w = factorization.V * Diagonal(factorization.S)^-1 * factorization.U' * q
            if isapprox(M*w, q; atol=atol, rtol=rtol)
                weights = zeros(datatype, length(differences))
                total = 1
                for (weight, count) in zip(w, counts[1:order])
                    weights[total:total+count-1] .= weight
                    total += count
                end
                return weights, differences
            end
            order += 1
        end
    end
    error("finitedifferences error: not compatible weights and difference vectors found.")
end

"""
    δ{T<:Real} <: Function

Delta function like Wannier functions.
"""
struct δ{T<:Real} <: Function
    unitary::Matrix{Complex{T}}
    coordinate::Vector{T}
    states::Vector{Int}
    function δ(unitary::AbstractMatrix, coordinate::AbstractVector, states::AbstractVector{Int})
        @assert length(states)==size(unitary)[2] "δ error: mismatched unitary and states."
        dtype = promote_type(typeof(real(first(unitary))), eltype(coordinate))
        new{dtype}(convert(Matrix{Complex{dtype}}, unitary), convert(Vector{dtype}, coordinate), convert(Vector{Int}, states))
    end
end
function (wannier::δ)(momentum::AbstractVector)
    result = copy(wannier.unitary)
    for (i, site) in enumerate(wannier.states)
       result[site, i] *= exp(-1im*dot(momentum, wannier.coordinate))
    end
    return result
end

"""
    MLWF{L<:Lattice, B<:BrillouinZone, T<:Real, V<:AbstractVector{T}} <: Function

Maximally localized Wannier functions on a Monkhorst-Pack mesh.
"""
struct MLWF{L<:Lattice, B<:BrillouinZone, T<:Real, V<:AbstractVector{T}} <: Function
    lattice::L
    brillouinzone::B
    eigenvalues::Vector{Vector{T}}
    eigenvectors::Vector{Matrix{Complex{T}}}
    unitaries::Vector{Matrix{Complex{T}}}
    weights::Vector{T}
    differences::Vector{V}
    neighbors::Vector{Vector{Int}}
    overlaps::Matrix{Matrix{Complex{T}}}
end
@inline Base.length(mlwf::MLWF) = length(mlwf.eigenvalues[1]) # number of Wannier functions
@inline Base.count(mlwf::MLWF) = length(mlwf.brillouinzone) # number of momenta
@inline dimension(mlwf::MLWF) = dimension(mlwf.lattice) # spatial dimension
@inline periods(mlwf::MLWF) = periods(keytype(mlwf.brillouinzone)) # periods of the Monkhorst-Pack mesh
@inline dtype(::MLWF{<:Lattice, <:BrillouinZone, T}) where {T<:Real} = T
@inline update!(mlwf::MLWF, method::FirstOrderOptimizer=GradientDescent(); kwargs...) = update!(mlwf, method; kwargs...)

"""
    update!(mlwf::MLWF, ::GradientDescent) -> MLWF

Update the maximally localized Wannier functions by the gradient descent method.
"""
function update!(mlwf::MLWF, ::GradientDescent)
    Ws = gradient(mlwf)
    b = - sum(mapreduce(abs2, +, W) for W in Ws)
    c = Ω₂(mlwf)
    ϵ = 1/4sum(mlwf.weights)
    backup = copy(mlwf.overlaps)
    while true
        ΔUs = [exp(ϵ .* W) for W in Ws]
        for i = 1:count(mlwf)
            for (j, index) in enumerate(mlwf.neighbors[i])
                mlwf.overlaps[j, i] = ΔUs[i]'*backup[j, i]*ΔUs[index]
            end
        end
        a = (Ω₂(mlwf)-b*ϵ-c)/ϵ^2
        if a>0
            ϵ = -b/2a
            ΔUs = [exp(ϵ .* W) for W in Ws]
            for i = 1:count(mlwf)
                for (j, index) in enumerate(mlwf.neighbors[i])
                    mlwf.overlaps[j, i] = ΔUs[i]'*backup[j, i]*ΔUs[index]
                end
                mlwf.unitaries[i] = mlwf.unitaries[i]*ΔUs[i]
            end
            break
        else
            ϵ = ϵ*0.8
        end
    end
    return mlwf
end

"""
    (mlwf::MLWF)(coordinate::AbstractVector) -> Matrix{Complex{dtype(mlwf)}}

Get the values of the wannier functions at each lattice point.
"""
function (mlwf::MLWF)(coordinate::AbstractVector)
    result = zeros(Complex{dtype(mlwf)}, length(mlwf.lattice), prod(periods(mlwf)), length(mlwf))
    for (j, index) in enumerate(product(map(i->-floor(Int, (i-1)/2):-floor(Int, (i-1)/2)+i-1, periods(mlwf))...))
        ΔR = mapreduce(*, +, index, mlwf.lattice.vectors) - coordinate
        for (n, momentum) in enumerate(mlwf.brillouinzone)
            phase = exp(1im*dot(ΔR, momentum))
            for k = 1:length(mlwf)
                for i = 1:length(mlwf.lattice)
                    result[i, j, k] += (mlwf.eigenvectors[n]*mlwf.unitaries[n])[i, k]*phase/count(mlwf)
                end
            end
        end
    end
    return reshape(result, (length(mlwf.lattice)*prod(periods(mlwf)), length(mlwf)))
end

"""
    rₙ(mlwf::MLWF) -> Vector{Vector{dtype(mlwf)}}

⟨rₙ⟩, i.e. the centers of the Wannier functions.
"""
function rₙ(mlwf::MLWF)
    result = Vector{dtype(mlwf)}[]
    @inbounds for n = 1:length(mlwf)
        r = zeros(dtype(mlwf), dimension(mlwf))
        for i = 1:count(mlwf)
            for (j, (weight, difference)) in enumerate(zip(mlwf.weights, mlwf.differences))
                r .+= -imag(log(mlwf.overlaps[j, i][n, n])) * weight /count(mlwf) .* difference
            end
        end
        push!(result, r)
    end
    return result
end

"""
    r²ₙ(mlwf::MLWF) -> Vector{dtype(mlwf)}

⟨r²ₙ⟩, i.e., the expectation value of r²ₙ for each Wannier function.
"""
function r²ₙ(mlwf::MLWF)
    result = dtype(mlwf)[]
    @inbounds for n = 1:length(mlwf)
        temp = 0.0
        for i = 1:count(mlwf)
            for (j, weight) in enumerate(mlwf.weights)
                temp += weight * (1-abs2(mlwf.overlaps[j, i][n, n])+imag(log(mlwf.overlaps[j, i][n, n]))^2)
            end
        end
        push!(result, temp/count(mlwf))
    end
    return result
end

"""
    Ω₁(mlwf::MLWF) -> dtype(mlwf)

The gauge-invariant part of the spread functional of the Wannier functions.
"""
function Ω₁(mlwf::MLWF)
    result = 0
    @inbounds for i = 1:count(mlwf)
        for (j, weight) in enumerate(mlwf.weights)
            result += weight * (length(mlwf)-sum(abs2, mlwf.overlaps[j, i]))
        end
    end
    return result/count(mlwf)
end

"""
    Ω₂(mlwf::MLWF) -> dtype(mlwf)

The gauge-dependent part of the spread functional of the Wannier functions.
"""
function Ω₂(mlwf::MLWF)
    centers = rₙ(mlwf)
    result = 0.0
    @inbounds for i = 1:count(mlwf)
        for (j, (weight, difference)) in enumerate(zip(mlwf.weights, mlwf.differences))
            temp = 0.0
            for m=1:length(mlwf), n=1:length(mlwf)
                temp += m==n ? (-imag(log(mlwf.overlaps[j, i][n, n]))-dot(difference, centers[n]))^2 : abs2(mlwf.overlaps[j, i][m, n])
            end
            result += temp*weight
        end
    end
    return result/count(mlwf)
end

"""
    Ω(mlwf::MLWF) -> dtype(mlwf)

The value of the spread functional of the Wannier functions.
"""
@inline Ω(mlwf::MLWF) = Ω₁(mlwf) + Ω₂(mlwf)

"""
    gradient(mlwf::MLWF) -> Vector{Matrix{Complex{dtype(mlwf)}}}

The gradient of the spread functional of the Wannier functions with respect to the infinitesimal gauge transformations at each momentum point.
"""
function gradient(mlwf::MLWF)
    centers = rₙ(mlwf)
    result = Matrix{Complex{dtype(mlwf)}}[]
    @inbounds for i = 1:count(mlwf)
        temp = zeros(Complex{dtype(mlwf)}, length(mlwf), length(mlwf))
        for (j, (weight, difference)) in enumerate(zip(mlwf.weights, mlwf.differences))
            M = mlwf.overlaps[j, i]
            D = diag(M)
            R = M * Diagonal(conj(D))
            q = Diagonal(imag(log.(D)) .+ map(center->dot(difference, center), centers))
            S = M * Diagonal(1 ./ D)
            T = S * q
            temp .+= 4 * weight .* ((R-R')/2-(T+T')/2im)
        end
        push!(result, temp)
    end
    return result
end

"""
    optimize!(mlwf::MLWF, method::FirstOrderOptimizer=GradientDescent(); maxiter::Int=20000, atol=atol, rtol=rtol, verbose::Union{Nothing, Int}=nothing) -> MLWF

Optimize a set of Wannier function to minimize the spread functional.
"""
function optimize!(mlwf::MLWF, method::FirstOrderOptimizer=GradientDescent(); maxiter::Int=20000, atol=atol, rtol=rtol, verbose::Union{Nothing, Int}=nothing)
    old, new = Ω₂(mlwf), 0.0
    for i = 1:maxiter
        update!(mlwf, method)
        new = Ω₂(mlwf)
        if !isnothing(verbose) && i%verbose==1
            println("|ΔΩ| = $(abs(new-old)) after $i iterations.")
        end
        if isapprox(old, new, atol=atol, rtol=rtol)
            isnothing(verbose) || println("Converged spread functional (Ω₂ = $new, |ΔΩ| = $(abs(new-old))) after $i iterations.")
            return mlwf
        end
        i<maxiter && (old = new)
    end
    @warn "optimize! warning: not converged spread functional after $maxiter iterations with |ΔΩ| = $(abs(new-old))."
    return mlwf
end

"""
    MLWF(tba::TBA, brillouinzone::BrillouinZone, bands::AbstractVector{Int}, guess::Union{δ, Nothing}=nothing; kwargs...)
    MLWF(hamiltonian::Function, lattice::Lattice, brillouinzone::BrillouinZone, bands::AbstractVector{Int}, guess::Union{δ, Nothing}=nothing; kwargs...)

Initialize a set of maximally localized Wannier functions.
"""
@inline function MLWF(tba::TBA, brillouinzone::BrillouinZone, bands::AbstractVector{Int}, guess::Union{δ, Nothing}=nothing; kwargs...)
    return MLWF(momentum::AbstractVector->matrix(tba; k=momentum), tba.lattice, brillouinzone, bands, guess; kwargs...)
end
function MLWF(hamiltonian::Function, lattice::Lattice, brillouinzone::BrillouinZone, bands::AbstractVector{Int}, guess::Union{δ, Nothing}=nothing; kwargs...)
    eigenvalues, eigenvectors, unitaries = Vector{dtype(lattice)}[], Matrix{Complex{dtype(lattice)}}[], Matrix{Complex{dtype(lattice)}}[]
    weights, differences = finitedifferences(map((reciprocal, N)->reciprocal/N, brillouinzone.reciprocals, periods(keytype(brillouinzone))); kwargs...)
    neighbors, overlaps = Vector{Int}[], Matrix{Matrix{Complex{dtype(lattice)}}}(undef, length(weights), length(brillouinzone))
    diffs = map(diff->keytype(brillouinzone)(diff, brillouinzone.reciprocals), differences)
    for momentum in keys(brillouinzone)
        k = expand(momentum, brillouinzone.reciprocals)
        eigensystem = eigen(hamiltonian(k))
        push!(eigenvalues, eigensystem.values[bands])
        push!(eigenvectors, eigensystem.vectors[:, bands])
        initials = isnothing(guess) ? eigenvectors[end] : guess(k)
        projection = eigenvectors[end]'*initials
        normalization = (projection'*projection)^(-1//2)
        push!(unitaries, projection*normalization)
        push!(neighbors, map(diff->Int(momentum+diff), diffs))
    end
    phases = [Diagonal([exp(-1im*dot(difference, lattice[k])) for k=1:length(lattice)]) for difference in differences]
    for i = 1:length(brillouinzone)
        left = (eigenvectors[i]*unitaries[i])'
        for j = 1:length(neighbors[i])
            right = eigenvectors[neighbors[i][j]]*unitaries[neighbors[i][j]]
            overlaps[j, i] = left*phases[j]*right
        end
    end
    return MLWF(lattice, brillouinzone, eigenvalues, eigenvectors, unitaries, weights, differences, neighbors, overlaps)
end

"""
    Hamiltonian{T<:Real, V<:AbstractVector{T}} <: Function

The Hamiltonian function based on Wannier interpolation.
"""
struct Hamiltonian{T<:Real, V<:AbstractVector{T}} <: Function
    num::Int
    coordinates::Vector{V}
    coefficients::Vector{Matrix{Complex{T}}}
end
@inline dtype(::Hamiltonian{T}) where {T<:Real} = T
@inline dimension(h::Hamiltonian) = h.num
function Hamiltonian(mlwf::MLWF; atol=atol, rtol=rtol)
    coordinates, coefficients = eltype(fieldtype(typeof(mlwf), :lattice))[], Matrix{Complex{dtype(mlwf)}}[]
    Hₖs = [U'*Diagonal(E)*U for (U, E) in zip(mlwf.unitaries, mlwf.eigenvalues)]
    for index in product(map(i->-floor(Int, (i-1)/2):-floor(Int, (i-1)/2)+i-1, periods(mlwf))...)
        R = mapreduce(*, +, index, mlwf.lattice.vectors)
        matrix = zeros(Complex{dtype(mlwf)}, length(mlwf), length(mlwf))
        for (Hₖ, momentum) in zip(Hₖs, mlwf.brillouinzone)
            phase = exp(-1im*dot(R, momentum))
            for m=1:length(mlwf), n=1:length(mlwf)
                matrix[m, n] += Hₖ[m, n]*phase/count(mlwf)
            end
        end
        if !isapprox(norm(matrix), 0; atol=atol, rtol=rtol)
            push!(coordinates, R)
            push!(coefficients, matrix)
        end
    end
    return Hamiltonian(length(mlwf), coordinates, coefficients)
end
function (h::Hamiltonian)(; k::AbstractVector, filter=coeff->true)
    result = zeros(Complex{dtype(h)}, dimension(h), dimension(h))
    for (R, coefficient) in zip(h.coordinates, h.coefficients)
        if filter(coefficient)
            phase = exp(1im*dot(R, k))
            for m=1:dimension(h), n=1:dimension(h)
                result[m, n] += phase*coefficient[m, n]
            end
        end
    end
    return result
end

end
