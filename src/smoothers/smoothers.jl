"""
    AbstractSmoother

Supertype of the smoothers, structs of hyperparameters; one left as `nothing` is chosen
from the data. `fit(model, y, t=eachindex(y))` returns a [`FittedSmoother`](@ref).
"""
abstract type AbstractSmoother end

"""
    AbstractTrendSmoother <: AbstractSmoother

Trend smoothers: [`LocalPolynomial`](@ref), [`MovingAverage`](@ref) and [`LOESS`](@ref).
"""
abstract type AbstractTrendSmoother <: AbstractSmoother end

"""
    AbstractSeasonalSmoother <: AbstractSmoother

Seasonal smoothers of known period, fitted with `fit(model, y, t; period)`:
[`SeasonalMovingAverage`](@ref) and [`StableSeasonal`](@ref).
"""
abstract type AbstractSeasonalSmoother <: AbstractSmoother end

"""
    FittedSmoother

A smoother fitted to `y` at sorted times `t`, with fields `model` (hyperparameters
resolved), `t`, `y`, `values` (fitted at `t`), `state` and `estimated` (the hyperparameters
chosen from the data and their criteria).
"""
struct FittedSmoother{M <: AbstractSmoother, S}
    model::M
    t::Vector{Float64}
    y::Vector{Float64}
    values::Vector{Float64}
    state::S
    estimated::NamedTuple
end

Base.getindex(f::FittedSmoother, name::Symbol) = getproperty(f, name)
Base.getindex(m::AbstractSmoother, name::Symbol) = getproperty(m, name)

function Base.show(io::IO, f::FittedSmoother)
    print(io, "FittedSmoother(", f.model, ", N=", length(f.t))
    isempty(f.estimated) || print(io, ", estimated: ", join(keys(f.estimated), ", "))
    return print(io, ")")
end

"""
    predict(f::FittedSmoother) -> Vector{Float64}
    predict(f::FittedSmoother, t_new; kwargs...) -> Vector{Float64}

Return the fitted values at `f.t`, or evaluate the fitted curve at the times `t_new`.
"""
predict(f::FittedSmoother) = f.values

fitted(f::FittedSmoother) = f.values
residuals(f::FittedSmoother) = f.y - f.values

# fit(Model, y, t; hyperparameters...) for fit(Model(; ...), y, t)
function fit(
        ::Type{M}, y::AbstractVector{<:Real}, t::AbstractVector{<:Real} = eachindex(y);
        kwargs...
) where {M <: AbstractTrendSmoother}
    return fit(M(; kwargs...), y, t)
end

function fit(
        model::AbstractSmoother,
        y::AbstractVector{<:Real},
        t::AbstractVector{<:Real} = eachindex(y);
        kwargs...
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
    Δ > 0 && all(i -> isapprox(t[i + 1] - t[i], Δ; rtol = 1e-8), 1:(length(t) - 1)) ||
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

function _check_in_range(t, t_new; atol::Real = 0.0)
    t₁, t_N = first(t), last(t)
    i = findfirst(x -> !(t₁ - atol ≤ x ≤ t_N + atol), t_new)
    isnothing(i) || throw(DomainError(t_new[i], "t_new must lie within [$t₁, $t_N]"))
    return nothing
end

# Fractional index of τ on the grid t₁, t₁ + Δ, …, snapped to an integer within 1e-8
function _grid_position(t₁::Real, Δ::Real, τ::Real)
    c = (τ - t₁) / Δ + 1
    return isapprox(c, round(c); atol = 1e-8) ? round(c) : c
end

# Report hyperparameters chosen from the data, in log group :estimated
function _log_estimated(context, estimated::NamedTuple)
    isempty(estimated) && return nothing
    @info "$context: hyperparameters chosen from the data" _group = :estimated estimated...
    return nothing
end

_name(m::AbstractSmoother) = nameof(typeof(m))
