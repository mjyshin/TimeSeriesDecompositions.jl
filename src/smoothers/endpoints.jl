"""
    AbstractEndpoints

Scheme for the asymmetric weights of a [`LocalPolynomial`](@ref) or
[`MovingAverage`](@ref) near the ends of a regularly spaced series, where the symmetric
weights do not fit: [`MinimumRevision`](@ref), [`DirectAsymmetric`](@ref) or
[`NoEndpoints`](@ref). Between grid points near the end, the same scheme is applied to
the grid extended beyond the data.
"""
abstract type AbstractEndpoints end

"""
    NoEndpoints()

No asymmetric weights: the first and last `h` points of the filtered series are `NaN`.
"""
struct NoEndpoints <: AbstractEndpoints end

"""
    DirectAsymmetric()

Direct asymmetric filter (DAF): at the end of the series, refit the same local polynomial
with the same kernel to the available observations (Proietti & Luati, 2008, eq. 4). It
reproduces polynomials of the filter's degree, but its real-time weights concentrate on
the last observation and give volatile estimates. Requires `h ≥ degree`; when
`h == degree` the fit is exactly determined and the real-time estimate is the last
observation itself.
"""
struct DirectAsymmetric <: AbstractEndpoints end

"""
    MinimumRevision(; degree=1, ratio=nothing)

Asymmetric weights minimising the mean square revision error with respect to the
symmetric filter (Proietti & Luati, 2008, §4), subject to reproducing polynomials of
degree `degree` (`d*`), assuming the signal also has a term `δ jᵈ*⁺¹` and white noise of
variance `σ²`.

- `degree`: `0` (LC: linear signal, constant fit), `1` (QL: quadratic signal, linear
  fit) or `2` (CQ: cubic signal, quadratic fit); must be below both the smoother's degree
  (`0` is allowed for a moving average) and `h`.
- `ratio`: `δ²/σ²`, a non-negative number or `Inf` (the limit in which degree `d* + 1`
  is also reproduced exactly, which needs `h > d* + 1`). With `nothing`, `fit` estimates
  it by minimising the mean square revision error of the real-time filter (P&L §5), and
  uses `0` if the series is too short for that. One ratio is used for every end point;
  an estimate of exactly `0` is common for noisy series.

Musgrave's end weights of X-11 are the LC case; see [`musgrave`](@ref).
"""
struct MinimumRevision{R<:Union{Nothing,Float64}} <: AbstractEndpoints
    degree::Int
    ratio::R
end

function MinimumRevision(; degree::Integer=1, ratio::Union{Nothing,Real}=nothing)
    degree ≥ 0 || throw(DomainError(degree, "degree must be non-negative"))
    isnothing(ratio) && return MinimumRevision(Int(degree), nothing)
    ratio ≥ 0 || throw(DomainError(ratio, "ratio must be non-negative"))
    return MinimumRevision(Int(degree), Float64(ratio))
end

"""
    musgrave(R) -> MinimumRevision

Musgrave (1964) end weights of X-11: [`MinimumRevision`](@ref) with `degree=0` and
`ratio = 4 / (π R²)`, where `R` is the irregular-to-trend (I/C) ratio. X-11 uses
`R = 1.0`, `3.5` and `4.5` with the 9-, 13- and 23-term Henderson filters; see
[`musgrave_ic_ratio`](@ref). Unlike the default QL weights, these do not reproduce a
linear trend at the end points.
"""
function musgrave(R::Real)
    R > 0 || throw(DomainError(R, "the I/C ratio R must be positive"))
    return MinimumRevision(; degree=0, ratio=4 / (π * R^2))
end

const _MUSGRAVE_IC_RATIOS = Dict(9 => 1.0, 13 => 3.5, 23 => 4.5)

"""
    musgrave_ic_ratio(terms) -> Float64

X-11's I/C ratio `R` for the Musgrave end weights of a Henderson filter of `terms`
samples (`bandwidth = (terms - 1)Δ`): `1.0` (9 terms), `3.5` (13 terms) or `4.5` (23
terms).
"""
function musgrave_ic_ratio(terms::Integer)
    haskey(_MUSGRAVE_IC_RATIOS, terms) ||
        throw(ArgumentError("X-11 defines I/C ratios for 9-, 13- and 23-term filters only"))
    return _MUSGRAVE_IC_RATIOS[terms]
end

Base.show(io::IO, ::NoEndpoints) = print(io, "NoEndpoints()")
Base.show(io::IO, ::DirectAsymmetric) = print(io, "DirectAsymmetric()")

function Base.show(io::IO, e::MinimumRevision)
    ratio = isnothing(e.ratio) ? "nothing" : round(e.ratio; sigdigits=4)
    return print(io, "MinimumRevision(degree=$(e.degree), ratio=$ratio)")
end
