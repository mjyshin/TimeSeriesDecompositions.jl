"""
    STLDecomp <: AbstractDecomp

Seasonal-trend decomposition with a loess trend, structured as [`X11Decomp`](@ref). Point
estimates only. See `fit(STLDecomp, ...)`.

!!! warning "Placeholder"
    This is not yet the STL procedure of Cleveland et al. (1990): it has no
    cycle-subseries smoothing, low-pass filter, inner/outer loop or robustness weighting.
"""
struct STLDecomp <: AbstractDecomp
    t::Vector{Float64}
    trend::Component
    seasonal::Component
    noise::Component
    rate::Component
    θ::NamedTuple
    smoother::NamedTuple
    _fit_info::NamedTuple  # (Δt, grid, estimated, derived, criteria)
end

const _STL_SMOOTHERS = (
    trend = (init = MovingAverage(), final = LOESS(; degree = 1)),
    seasonal = (final = StableSeasonal(),)
)

_stages(::STLDecomp) = ((:trend, :init), (:seasonal, :final), (:trend, :final))

"""
    fit(STLDecomp, y, t=eachindex(y); period, smoother=(;)) -> STLDecomp

Loess-based decomposition of `y` at uniformly spaced times `t`: an initial trend
(`trend.init`, default [`MovingAverage`](@ref)`()` over one period) is removed, a seasonal
filter (`seasonal.final`, default [`StableSeasonal`](@ref)`()`) of the detrended series
gives the seasonal, and a trend smoother (`trend.final`, default local-linear
[`LOESS`](@ref)) of the seasonally adjusted series gives the trend. `smoother` and the
reported hyperparameters are as for [`X11Decomp`](@ref); a loess span left as `nothing`
covers the smallest odd number `≥ 1.5p / (1 - 1.5 / 7)` of observations [1].

[1] R. B. Cleveland, W. S. Cleveland, J. E. McRae and I. Terpenning. "STL: a seasonal-trend
decomposition procedure based on loess". Journal of Official Statistics 6 (1), 1990.
"""
function fit(
        ::Type{STLDecomp},
        y::AbstractVector{<:Real},
        t::AbstractVector{<:Real} = eachindex(y);
        period::Real,
        smoother::NamedTuple = (;)
)
    y, t, Δt = _uniform_grid(y, t)
    p = _samples_per_period(period, Δt)
    N = length(y)
    N ≥ 2p || throw(ArgumentError("need at least two periods of observations"))
    s = _resolve_smoothers(smoother, _STL_SMOOTHERS)
    initial, derived_init = _resolve_trend(s.trend.init, p, Δt, N)
    final, derived_final = _resolve_stl_trend(s.trend.final, p, Δt, N)
    r = _Report(period)

    trend_init = _fit(initial, y, t)
    _record!(r, trend_init, :_trend_init; derived = keys(derived_init))
    seasonal_final, seasonal, note = _seasonal(
        s.seasonal.final, y - predict(trend_init), t, p, Δt
    )
    _record!(r, seasonal_final, :_seasonal_final)
    isnothing(note) || push!(r.criteria, :note_seasonal_final => note)
    adjusted = y - seasonal
    trend_final = _fit(final, adjusted, t)
    _record!(r, trend_final, :_trend_final; derived = keys(derived_final))
    trend_values = predict(trend_final)

    noise = adjusted - trend_values
    rate = _trend_rate(trend_values, Δt)
    grid = (; t, trend = trend_values, seasonal, noise, rate)
    θ, fit_info = _finish(r, STLDecomp, Δt, grid)
    fitted = (trend = (init = trend_init, final = trend_final),
        seasonal = (final = seasonal_final,))
    return STLDecomp(
        t,
        Component(trend_values),
        Component(seasonal),
        Component(noise),
        Component(rate),
        θ,
        fitted,
        fit_info
    )
end

# Cleveland et al.'s trend window n_t for a loess span left as nothing
function _resolve_stl_trend(m::LOESS, p, _, N)
    isnothing(m.span) && isnothing(m.bandwidth) || return m, (;)
    q = _nextodd(1.5p / (1 - 1.5 / 7))
    span = q / N
    return LOESS(span, nothing, m.degree), (; span, neighbors = q)
end

_resolve_stl_trend(m::AbstractTrendSmoother, p, Δt, N) = _resolve_trend(m, p, Δt, N)
