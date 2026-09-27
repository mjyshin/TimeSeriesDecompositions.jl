"""
    AbstractSmoother

Supertype of the smoothers: structs holding hyperparameters only. A hyperparameter left
as `nothing` is chosen from the data when the smoother is fitted.

`fit(model, y, t=eachindex(y))` fits the smoother to observations `y` at times `t` and
returns a [`FittedSmoother`](@ref); `predict(f)` gives the fitted values at `t` and
`predict(f, t_new)` evaluates the fitted curve at other times. To smooth another series
with the same (resolved) hyperparameters, refit `f.model`.
"""
abstract type AbstractSmoother end

"""
    AbstractTrendSmoother <: AbstractSmoother

Smoothers of a trend: [`LocalPolynomial`](@ref) (e.g. [`Henderson`](@ref)),
[`MovingAverage`](@ref) and [`LOESS`](@ref).
"""
abstract type AbstractTrendSmoother <: AbstractSmoother end

"""
    AbstractSeasonalSmoother <: AbstractSmoother

Smoothers of a seasonal pattern with a known period: [`SeasonalMovingAverage`](@ref)
and [`StableSeasonal`](@ref). Fitted with `fit(model, y, t; period)`.
"""
abstract type AbstractSeasonalSmoother <: AbstractSmoother end

"""
    FittedSmoother

A smoother fitted to observations `y` at times `t` (sorted). Fields: `model` (the
smoother with every hyperparameter resolved), `t`, `y`, `values` (the fitted values at
`t`), `state` (method-specific precomputations) and `estimated` (the hyperparameters
chosen from the data, with the criteria that chose them).
"""
struct FittedSmoother{M<:AbstractSmoother,S}
    model::M
    t::Vector{Float64}
    y::Vector{Float64}
    values::Vector{Float64}
    state::S
    estimated::NamedTuple
end

function Base.show(io::IO, f::FittedSmoother)
    print(io, "FittedSmoother(", f.model, ", N=", length(f.t))
    isempty(f.estimated) || print(io, ", estimated: ", join(keys(f.estimated), ", "))
    return print(io, ")")
end

"""
    predict(f::FittedSmoother) -> Vector{Float64}
    predict(f::FittedSmoother, t_new; kwargs...) -> Vector{Float64}

Fitted values at the observed times `f.t`, or the fitted curve evaluated at the times
`t_new` (see the smoother's documentation for which times are allowed).
"""
predict(f::FittedSmoother) = f.values

fitted(f::FittedSmoother) = f.values
residuals(f::FittedSmoother) = f.y - f.values

# `fit(Model, y, t; hyperparameters...)` as shorthand for `fit(Model(; ...), y, t)`
function fit(
    ::Type{M}, y::AbstractVector{<:Real}, t::AbstractVector{<:Real}=eachindex(y); kwargs...
) where {M<:AbstractTrendSmoother}
    return fit(M(; kwargs...), y, t)
end

function fit(
    model::AbstractSmoother,
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    kwargs...,
)
    f = _fit(model, y, t; kwargs...)
    _log_estimated(_name(model), f.estimated)
    return f
end

function _check_series(y::AbstractVector{<:Real})
    all(isfinite, y) ||
        throw(ArgumentError("y must be finite (missing values are not supported)"))
    return collect(Float64, y)
end

# Sorted (y, t) as Float64 vectors
function _sorted_data(y::AbstractVector{<:Real}, t::AbstractVector{<:Real})
    y = _check_series(y)
    length(t) == length(y) || throw(DimensionMismatch("y and t must have the same length"))
    all(isfinite, t) || throw(ArgumentError("t must be finite"))
    perm = sortperm(t)
    return y[perm], Float64.(t[perm])
end

# Spacing of uniformly spaced, sorted t, or nothing
function _spacing(t::AbstractVector{Float64})
    length(t) ≥ 2 || return nothing
    Δ = t[2] - t[1]
    Δ > 0 && all(i -> isapprox(t[i + 1] - t[i], Δ; rtol=1e-8), 1:(length(t) - 1)) ||
        return nothing
    return Δ
end

function _require_grid(t, name)
    Δ = _spacing(t)
    isnothing(Δ) && throw(
        ArgumentError("$name requires uniformly spaced t; use LOESS for irregular times"),
    )
    return Δ
end

function _check_in_range(t, t_new)
    t₁, t_N = first(t), last(t)
    i = findfirst(x -> !(t₁ ≤ x ≤ t_N), t_new)
    isnothing(i) || throw(DomainError(t_new[i], "t_new must lie within [$t₁, $t_N]"))
    return nothing
end

# Report hyperparameters that were chosen from the data. Silence with the Logging stdlib,
# e.g. by filtering on `group == :estimated`.
function _log_estimated(context, estimated::NamedTuple)
    isempty(estimated) && return nothing
    @info "$context: hyperparameters chosen from the data" _group = :estimated estimated...
    return nothing
end

_nothing_or(x) = something(x, "nothing")

# Name of a smoother in messages
_name(m::AbstractSmoother) = nameof(typeof(m))
