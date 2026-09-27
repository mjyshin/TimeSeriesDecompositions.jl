"""
    AbstractFilter

Supertype of the linear filters used by the classical decompositions.

Every filter `F` is created from data with `fit(F, y; hyperparameters...)`, which
chooses any hyperparameters left to the data (reported with an `@info` message and
recorded in the filter) and stores the resulting weights. `predict(f, y)` applies the
fitted filter to a series `y` (the same series or a new one), returning a vector of the
same length.
"""
abstract type AbstractFilter end

"""
    AbstractTrendFilter <: AbstractFilter

Filters that extract a smooth trend: [`LocalPolynomialFilter`](@ref) and
[`MovingAverage`](@ref). See also [`LoessSmoother`](@ref) for irregular times.
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

# Report hyperparameters that were chosen from the data. Silence with the Logging stdlib,
# e.g. by filtering on `group == :estimated`.
function _log_estimated(context, estimated::NamedTuple)
    isempty(estimated) && return nothing
    @info "$context: hyperparameters chosen from the data" _group = :estimated estimated...
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

# Centred filter with weights w (odd length) where the full window fits; returns the
# indices of those points and the filtered values at them
function _symmetric_filter(w::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    h = (length(w) - 1) ÷ 2
    inner = (h + 1):(length(y) - h)
    return inner, [dot(w, view(y, (t - h):(t + h))) for t in inner]
end
