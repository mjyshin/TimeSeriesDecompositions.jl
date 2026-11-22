"""
    STL(; seasonal_final, trend_final, iterations=2)
    STL(; seasonal, trend, iterations=2)

STL decomposition model of Cleveland et al. [1], without the robustness weights of its
outer loop. Each keyword is the smoother of one stage, listed in fitting order; `seasonal`
and `trend` are the same stages under their component names.

- `seasonal_final`: [`SeasonalLOESS`](@ref)`()`, with its bandwidth chosen in each pass by
  X-11's moving seasonality ratio of that pass's detrended series, as in [`X11`](@ref).
- `trend_final`: local-quadratic [`LOESS`](@ref), whose span is chosen by leave-one-out
  cross-validation from the smallest odd number `≥ 1.5p / (1 - 1.5 / 7)` of observations
  up, the trend window that [1] gives for a seven-cycle seasonal window. [1] fits the trend
  locally linear, which flattens peaks and troughs; `LOESS(; degree=1)` restores it.
- `iterations`: passes of seasonal then trend smoothing (the inner loop of [1]).

[1] R. B. Cleveland, W. S. Cleveland, J. E. McRae and I. Terpenning. "STL: a seasonal-trend
decomposition procedure based on loess". Journal of Official Statistics 6 (1), 1990.
"""
struct STL <: AbstractDecompositionModel
    seasonal_final::AbstractSeasonalSmoother
    trend_final::AbstractTrendSmoother
    iterations::Int
    function STL(seasonal_final, trend_final, iterations::Integer)
        s = _check_stages(STL, (; seasonal_final, trend_final))
        iterations ≥ 1 || throw(DomainError(iterations, "iterations must be at least 1"))
        return new(s..., iterations)
    end
end

const _STL_DEFAULTS = (
    seasonal_final = SeasonalLOESS(),
    trend_final = LOESS()
)

function STL(; iterations::Integer = 2, kwargs...)
    return STL(_stage_settings(STL, _STL_DEFAULTS, NamedTuple(kwargs))..., iterations)
end

"""
    fit(model::STL, y, t=eachindex(y); period) -> Decomposition{STL}

Decompose `y` at uniformly spaced times `t` with `p = period / Δt` (an integer) samples per
cycle and at least two cycles. From a zero trend, each of the `iterations` passes

 1. smooths each cycle-subseries of the detrended series (`seasonal_final`), and one cycle
    beyond each end;
 2. takes the seasonal as that less its low-pass filter: moving averages of `p`, `p` and 3
    terms, then a local-linear [`LOESS`](@ref) with the smallest odd number `≥ p` of
    neighbors (none for `p ≤ 3`, where it would reproduce its input);
 3. smooths the seasonally adjusted series for the trend (`trend_final`).

The first pass chooses the seasonal bandwidth from the series itself, as its trend is zero,
so a single pass tends to a short window. A seasonal smoother other than
[`SeasonalLOESS`](@ref) is extended by the nearest value of
the same phase. The outputs are as for `fit(::X11, ...)`, with the smoothers of the last
pass.
"""
function fit(model::STL, y::AbstractVector{<:Real},
        t::AbstractVector{<:Real} = eachindex(y); period::Real)
    y, t, Δt = _uniform_grid(y, t)
    p = _samples_per_period(period, Δt)
    N = length(y)
    N ≥ 2p || throw(ArgumentError("need at least two periods of observations"))
    final, derived_final = _resolve_stl_trend(model.trend_final, p, Δt, N)

    trend = zeros(N)
    local seasonal_final, seasonal, note, trend_final, derived, chosen, msr
    for _ in 1:(model.iterations)
        detrended = y - trend
        seasonal_model, derived, chosen, msr = _final_seasonal(
            model.seasonal_final, detrended, p, p * Δt)
        seasonal_final, seasonal, note = _stl_seasonal(
            seasonal_model, detrended, t, p, Δt)
        trend_final = _fit(final, y - seasonal, t)
        trend = predict(trend_final)
    end

    r = _Report()
    append!(r.criteria, msr)
    _record!(r, seasonal_final, :seasonal_final; derived, chosen)
    isnothing(note) || push!(r.criteria, :note_seasonal_final => note)
    _record!(r, trend_final, :trend_final; derived = keys(derived_final))
    smoother = (; seasonal_final, trend_final)
    return _filter_decomposition(
        model, r, y, t, Δt, period, trend, seasonal, smoother)
end

# Candidate loess spans from the trend window of Cleveland et al. for a seven-cycle seasonal
function _resolve_stl_trend(m::LOESS, p, _, N)
    _loess_candidates(m, _nextodd(1.5p / (1 - 1.5 / 7)), N)
end

_resolve_stl_trend(m::AbstractTrendSmoother, p, Δt, N) = _resolve_trend(m, p, Δt, N)

# Fitted smoother, seasonal and a note (if the filter was shortened) of the detrended series
# x, NaN where the trend is undefined: the cycle-subseries smoothing C, one cycle beyond
# each end of where x is defined and extended to N + 2p values by the nearest value of the
# same phase, less its low-pass filter
function _stl_seasonal(model::AbstractSeasonalSmoother, x, t, p, Δt)
    N = length(x)
    inner = _finite_range(x)
    model, note = _fitting_seasonal(model, length(inner) ÷ p)
    fitted = _fit(model, x[inner], t[inner]; period = p * Δt)
    c = fill(NaN, N + 2p)    # at indices 1 - p, …, N + p
    c[(first(inner):(last(inner) + 2p))] = _extended_seasonal(
        fitted.model, x[inner], p, p * Δt)
    for j in (first(inner) - 1):-1:1
        c[j] = c[j + p]
    end
    for j in (last(inner) + 2p + 1):(N + 2p)
        c[j] = c[j - p]
    end
    return fitted, c[(p + 1):(p + N)] - _low_pass(c, p), note
end

# Low-pass filter of STL of the N + 2p values c: N values at the middle indices. A loess
# of 3 neighbors (p ≤ 3) gives the centers zero weight and is skipped, as STL reproduces x
function _low_pass(c, p)
    x = _moving_average(_moving_average(_moving_average(c, p), p), 3)
    N, q = length(x), _nextodd(p)
    q > 3 || return x
    return first(_loess_fit(collect(1.0:N), x, ones(N), q / N, 0.0, 1))
end

# The (length(x) - k + 1) means of k consecutive values
function _moving_average(x, k)
    return [mean(view(x, i:(i + k - 1))) for i in 1:(length(x) - k + 1)]
end
