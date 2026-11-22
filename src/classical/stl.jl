"""
    STL(; trend_init, seasonal_final, trend_final)
    STL(; trend, seasonal)

Seasonal-trend decomposition model with a loess trend, structured as [`X11`](@ref) with one
pass. Each keyword is the smoother of one stage, listed in fitting order; `trend` sets both
trend stages.

- `trend_init`: [`MovingAverage`](@ref)`()` over one period.
- `seasonal_final`: [`StableSeasonal`](@ref)`()`.
- `trend_final`: local-linear [`LOESS`](@ref); a span left as `nothing` is chosen by
  leave-one-out cross-validation from the smallest odd number `≥ 1.5p / (1 - 1.5 / 7)` of
  observations [1] up.

!!! warning "Placeholder"
    This is not yet the STL procedure of Cleveland et al. [1]: it has no cycle-subseries
    smoothing, low-pass filter, inner/outer loop or robustness weighting.

[1] R. B. Cleveland, W. S. Cleveland, J. E. McRae and I. Terpenning. "STL: a seasonal-trend
decomposition procedure based on loess". Journal of Official Statistics 6 (1), 1990.
"""
struct STL <: AbstractDecompositionModel
    trend_init::AbstractTrendSmoother
    seasonal_final::AbstractSeasonalSmoother
    trend_final::AbstractTrendSmoother
    function STL(trend_init, seasonal_final, trend_final)
        s = (; trend_init, seasonal_final, trend_final)
        return new(_check_stages(STL, s)...)
    end
end

const _STL_DEFAULTS = (
    trend_init = MovingAverage(),
    seasonal_final = StableSeasonal(),
    trend_final = LOESS(; degree = 1)
)

STL(; kwargs...) = STL(_stage_settings(STL, _STL_DEFAULTS, NamedTuple(kwargs))...)

"""
    fit(model::STL, y, t=eachindex(y); period) -> Decomposition{STL}

Decompose `y` at uniformly spaced times `t` with at least two cycles: the initial trend
(`trend_init`) is removed, the seasonal filter (`seasonal_final`) of the detrended series
gives the seasonal, and the trend smoother (`trend_final`) of the seasonally adjusted series
gives the trend. The outputs are as for `fit(::X11, ...)`.
"""
function fit(model::STL, y::AbstractVector{<:Real},
        t::AbstractVector{<:Real} = eachindex(y); period::Real)
    y, t, Δt = _uniform_grid(y, t)
    p = _samples_per_period(period, Δt)
    N = length(y)
    N ≥ 2p || throw(ArgumentError("need at least two periods of observations"))
    initial, derived_init = _resolve_trend(model.trend_init, p, Δt, N)
    final, derived_final = _resolve_stl_trend(model.trend_final, p, Δt, N)
    r = _Report()

    trend_init = _fit(initial, y, t)
    _record!(r, trend_init, :trend_init; derived = keys(derived_init))
    seasonal_final, seasonal, note = _seasonal(
        model.seasonal_final, y - predict(trend_init), t, p, Δt)
    _record!(r, seasonal_final, :seasonal_final)
    isnothing(note) || push!(r.criteria, :note_seasonal_final => note)
    trend_final = _fit(final, y - seasonal, t)
    _record!(r, trend_final, :trend_final; derived = keys(derived_final))

    smoother = (; trend_init, seasonal_final, trend_final)
    return _filter_decomposition(
        model, r, y, t, Δt, period, predict(trend_final), seasonal,
        smoother)
end

# Candidate loess spans from Cleveland et al.'s trend window n_t
function _resolve_stl_trend(m::LOESS, p, _, N)
    _loess_candidates(m, _nextodd(1.5p / (1 - 1.5 / 7)), N)
end

_resolve_stl_trend(m::AbstractTrendSmoother, p, Δt, N) = _resolve_trend(m, p, Δt, N)
