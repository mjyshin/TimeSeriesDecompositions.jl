"""
    AbstractEndpoints

Asymmetric weights of a [`LocalPolynomial`](@ref) or [`MovingAverage`](@ref) at the ends of
a regularly spaced series: [`MinimumRevision`](@ref), [`DirectAsymmetric`](@ref) or
[`NoEndpoints`](@ref).
"""
abstract type AbstractEndpoints end

Base.getindex(e::AbstractEndpoints, name::Symbol) = getproperty(e, name)

"""
    NoEndpoints()

No asymmetric weights: the first and last `h` points of the filtered series are `NaN`.
"""
struct NoEndpoints <: AbstractEndpoints end

"""
    DirectAsymmetric()

Direct asymmetric filter: the same local polynomial and kernel fitted to the available
observations (Proietti & Luati, 2008, eq. 4). Requires `h ≥ degree`.
"""
struct DirectAsymmetric <: AbstractEndpoints end

"""
    MinimumRevision(; degree=1, ratio=nothing)

Asymmetric weights minimizing the mean square revision error against the symmetric filter
[1, §4], reproducing polynomials of degree `degree` for a signal with an extra term `δ jᵈ⁺¹`
in white noise of variance `σ²`.

- `degree`: `0` (LC), `1` (QL) or `2` (CQ); below both the smoother's degree (`0` is
  allowed for a moving average) and `h`.
- `ratio`: `δ²/σ²`, non-negative or `Inf` (degree `d + 1` also reproduced, needs
  `h > d + 1`). `nothing` estimates one ratio for every end point by the mean square
  revision error of the real-time filter [1, §5], or uses `0` for short series.

Musgrave's end weights of X-11 are the LC case; see [`musgrave`](@ref).

[1] T. Proietti and A. Luati. "Real time estimation in local polynomial regression, with
application to trend-cycle analysis". Annals of Applied Statistics 2 (4), 2008.
"""
struct MinimumRevision{R <: Union{Nothing, Float64}} <: AbstractEndpoints
    degree::Int
    ratio::R
end

function MinimumRevision(; degree::Integer = 1, ratio::Union{Nothing, Real} = nothing)
    degree ≥ 0 || throw(DomainError(degree, "degree must be non-negative"))
    isnothing(ratio) && return MinimumRevision(Int(degree), nothing)
    ratio ≥ 0 || throw(DomainError(ratio, "ratio must be non-negative"))
    return MinimumRevision(Int(degree), Float64(ratio))
end

"""
    musgrave(R) -> MinimumRevision

Musgrave (1964) end weights of X-11, [`MinimumRevision`](@ref) with `degree=0` and
`ratio = 4 / (π R²)` for the irregular-to-trend ratio `R` (see [`musgrave_ic_ratio`](@ref)).
"""
function musgrave(R::Real)
    R > 0 || throw(DomainError(R, "the I/C ratio R must be positive"))
    return MinimumRevision(; degree = 0, ratio = 4 / (π * R^2))
end

const _MUSGRAVE_IC_RATIOS = Dict(9 => 1.0, 13 => 3.5, 23 => 4.5)

"""
    musgrave_ic_ratio(terms) -> Float64

Return X-11's I/C ratio for a Henderson filter of `terms` samples: `1.0` (9), `3.5` (13) or
`4.5` (23).
"""
function musgrave_ic_ratio(terms::Integer)
    haskey(_MUSGRAVE_IC_RATIOS, terms) ||
        throw(ArgumentError("X-11 defines I/C ratios for 9-, 13- and 23-term filters only"))
    return _MUSGRAVE_IC_RATIOS[terms]
end

Base.show(io::IO, ::NoEndpoints) = print(io, "NoEndpoints()")
Base.show(io::IO, ::DirectAsymmetric) = print(io, "DirectAsymmetric()")

function Base.show(io::IO, e::MinimumRevision)
    ratio = isnothing(e.ratio) ? "nothing" : round(e.ratio; sigdigits = 4)
    return print(io, "MinimumRevision(degree=$(e.degree), ratio=$ratio)")
end
