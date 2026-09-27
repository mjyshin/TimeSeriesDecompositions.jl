"""
    AbstractFilter

Supertype of the linear filters used by the classical decompositions.

Every filter `F` is created from data with `fit(F, y; hyperparameters...)`, which
estimates any data-driven hyperparameters (e.g. the minimum-revision `ratio`) and stores
the resulting weights. `predict(f, y)` applies the fitted filter to a series `y`
(the same series or a new one), returning a vector of the same length.
"""
abstract type AbstractFilter end

"""
    AbstractTrendFilter <: AbstractFilter

Filters that extract a smooth trend: [`LocalPolynomialFilter`](@ref),
[`MovingAverage`](@ref) and [`LoessFilter`](@ref).
"""
abstract type AbstractTrendFilter <: AbstractFilter end

"""
    AbstractSeasonalFilter <: AbstractFilter

Filters that extract a seasonal pattern from a detrended series:
[`StableSeasonalFilter`](@ref) and [`SeasonalMovingAverage`](@ref).
"""
abstract type AbstractSeasonalFilter <: AbstractFilter end

function _check_series(y::AbstractVector{<:Real})
    all(isfinite, y) ||
        throw(ArgumentError("y must be finite (missing values are not supported)"))
    return collect(Float64, y)
end

function _check_length(y::AbstractVector, window::Integer)
    length(y) ≥ window || throw(
        ArgumentError("series of length $(length(y)) is shorter than the window $window"),
    )
    return nothing
end

# Centred filter with weights w (odd length); at the ends, the weights falling outside
# the series are dropped and the remainder renormalised to sum to one
function _truncated_filter(w::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    h = (length(w) - 1) ÷ 2
    N = length(y)
    return map(1:N) do t
        js = max(-h, 1 - t):min(h, N - t)
        wt = view(w, js .+ (h + 1))
        return dot(wt, view(y, t .+ js)) / sum(wt)
    end
end
