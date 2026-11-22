"""
    X11(; trend_init, seasonal_init, trend_interm, seasonal_final, trend_final)
    X11(; trend, seasonal)

X-11-style decomposition model: trend smoothers and `S(n × m)` seasonal filters in two
passes, for any integer period. Each keyword is the smoother of one stage, listed in fitting
order; `trend` or `seasonal` sets every stage of that component.

- `trend_init`: [`MovingAverage`](@ref)`()` over one period.
- `seasonal_init`: [`SeasonalMovingAverage`](@ref)`(n=3, m=3)`.
- `trend_interm`: `nothing`, which is `trend_final` with minimum-revision ratio `0`.
- `seasonal_final`: `nothing`, `S3×3`, `S3×5` or `S3×9` by X-11's moving seasonality
  ratio (`S3×3` below seven cycles).
- `trend_final`: [`Henderson`](@ref)`()`, with its length chosen by cross-validation and
  estimated QL minimum-revision end weights.

```julia
fit(X11(; trend_interm=Henderson(; bandwidth=12, endpoints=MinimumRevision(; ratio=0))),
    y; period=7)
fit(X11(; trend_final=Henderson(; bandwidth=12, endpoints=musgrave(3.5))), y; period=12)
```
"""
struct X11 <: AbstractDecompositionModel
    trend_init::AbstractTrendSmoother
    seasonal_init::AbstractSeasonalSmoother
    trend_interm::Union{Nothing, AbstractTrendSmoother}
    seasonal_final::Union{Nothing, AbstractSeasonalSmoother}
    trend_final::AbstractTrendSmoother
    function X11(trend_init, seasonal_init, trend_interm, seasonal_final, trend_final)
        s = (; trend_init, seasonal_init, trend_interm, seasonal_final, trend_final)
        return new(_check_stages(X11, s)...)
    end
end

const _X11_DEFAULTS = (
    trend_init = MovingAverage(),
    seasonal_init = SeasonalMovingAverage(; n = 3, m = 3),
    trend_interm = nothing,
    seasonal_final = nothing,
    trend_final = Henderson()
)

X11(; kwargs...) = X11(_stage_settings(X11, _X11_DEFAULTS, NamedTuple(kwargs))...)

"""
    fit(model::X11, y, t=eachindex(y); period) -> Decomposition{X11}

Decompose `y` at uniformly spaced times `t` with `p = period / Δt` (an integer) samples per
cycle and at least three cycles, in two passes:

 1. initial trend (`trend_init`); initial seasonal (`seasonal_init`) of the detrended
    series; intermediate trend (`trend_interm`) of the seasonally adjusted series;
 2. seasonal (`seasonal_final`) of the series detrended by the intermediate trend; trend
    (`trend_final`) of the seasonally adjusted series.

Each seasonal filter is applied where the trend is defined, extended to the ends by the
nearest estimate of the same phase, and centered by its `2 × p` moving average. The noise is
`y - trend - seasonal`. With the default smoothers, a linear trend plus a fixed seasonal
pattern is reproduced exactly.

A trend bandwidth or span left as `nothing` is chosen by leave-one-out cross-validation on
the series that stage smooths, from a minimum set by `p` so that the trend cannot absorb
the seasonal: a [`LocalPolynomial`](@ref) bandwidth `2hΔt` from the smallest even
`2h ≥ p` (at least 4), and a [`LOESS`](@ref) span from the smallest odd number `≥ p + 3`
(at least 7) of observations. A [`MovingAverage`](@ref) bandwidth left as `nothing` is one
period. A seasonal filter needing more cycles than
available is replaced by the longest that fits, with a note.

`d.smoother` holds the fitted smoother of each stage, and `d.θ` their numeric
hyperparameters, named `<quantity>_<stage>` in fitting order (e.g. `bandwidth_trend_final`);
`d._fit_info` lists those `estimated` from the data and `derived` from the period, and the
`criteria` and notes. The rate is the derivative of the not-a-knot cubic spline through the
trend. There is no extreme-value, calendar or forecast adjustment as in X-13ARIMA-SEATS.
"""
function fit(model::X11, y::AbstractVector{<:Real},
        t::AbstractVector{<:Real} = eachindex(y); period::Real)
    y, t, Δt = _uniform_grid(y, t)
    p = _samples_per_period(period, Δt)
    N = length(y)
    cycles = N ÷ p
    cycles ≥ 3 || throw(ArgumentError("need at least 3 seasonal cycles, got $cycles"))
    initial, derived_init = _resolve_trend(model.trend_init, p, Δt, N)
    interm = something(model.trend_interm, _interm(model.trend_final))
    interm, derived_interm = _resolve_trend(interm, p, Δt, N)
    final, derived_final = _resolve_trend(model.trend_final, p, Δt, N)
    r = _Report()

    # Pass 1
    trend_init = _fit(initial, y, t)
    _record!(r, trend_init, :trend_init; derived = keys(derived_init))
    seasonal_init, seasonal_values, note = _seasonal(
        model.seasonal_init, y - predict(trend_init), t, p, Δt)
    _record!(r, seasonal_init, :seasonal_init)
    isnothing(note) || push!(r.criteria, :note_seasonal_init => note)
    trend_interm = _fit(interm, y - seasonal_values, t)
    _record!(r, trend_interm, :trend_interm; derived = keys(derived_interm))

    # Pass 2
    detrended = y - predict(trend_interm)
    inner = _finite_range(detrended)
    seasonal_model, derived, chosen = model.seasonal_final, (), ()
    if isnothing(seasonal_model)
        available = length(inner) ÷ p
        if available < _MSR_MIN_CYCLES
            note = "MSR needs $_MSR_MIN_CYCLES cycles, got $available; using S3×3"
            push!(r.criteria, :note_msr => note)
            seasonal_model, derived = SeasonalMovingAverage(; n = 3, m = 3), (:n, :m)
        else
            seasonal_model, msr = _msr_filter(detrended[inner], p)
            push!(r.criteria, :msr => msr)
            chosen = (:n, :m)
        end
    end
    seasonal_final, seasonal_values, note = _seasonal(seasonal_model, detrended, t, p, Δt)
    _record!(r, seasonal_final, :seasonal_final; derived, chosen)
    isnothing(note) || push!(r.criteria, :note_seasonal_final => note)
    adjusted = y - seasonal_values
    trend_final = _fit(final, adjusted, t)
    _record!(r, trend_final, :trend_final; derived = keys(derived_final))

    smoother = (; trend_init, seasonal_init, trend_interm, seasonal_final, trend_final)
    return _filter_decomposition(model, r, y, t, Δt, period, predict(trend_final),
        seasonal_values, smoother)
end

# Trend smoother with hyperparameters left as nothing resolved from p: candidates from a
# minimum, or a value derived from p (and the derived values)
function _resolve_trend(m::LocalPolynomial, p, Δt, N)
    isnothing(m.bandwidth) || return m, (;)
    h_min = max(2, cld(p, 2), _min_halfwidth(m.degree))
    bandwidth = _bandwidth_grid(h_min, N, Δt)
    return LocalPolynomial(bandwidth, m.degree, m.kernel, m.endpoints), (;)
end

function _resolve_trend(m::LOESS, p, _, N)
    return _loess_candidates(m, max(7, _nextodd(p + 3)), N)
end

function _loess_candidates(m::LOESS, q_min, N)
    isnothing(m.span) && isnothing(m.bandwidth) || return m, (;)
    return LOESS(_neighbor_grid(q_min, N) ./ N, nothing, m.degree), (;)
end

function _resolve_trend(m::MovingAverage, p, Δt, _)
    isnothing(m.bandwidth) || return m, (;)
    bandwidth = p * Δt
    return MovingAverage(bandwidth, m.endpoints, m.gapfactor), (; bandwidth)
end

_interm(m::AbstractTrendSmoother) = m

function _interm(m::LocalPolynomial)
    return LocalPolynomial(m.bandwidth, m.degree, m.kernel, _interm(m.endpoints))
end

function _interm(m::MovingAverage)
    return MovingAverage(m.bandwidth, _interm(m.endpoints), m.gapfactor)
end

_interm(e::AbstractEndpoints) = e
_interm(e::MinimumRevision) = MinimumRevision(e.degree, 0.0)

# Indices of the finite values of x, which must be contiguous
function _finite_range(x)
    idx = findall(isfinite, x)
    isempty(idx) && throw(ArgumentError("the trend is undefined everywhere"))
    inner = first(idx):last(idx)
    length(idx) == length(inner) || throw(ArgumentError("the trend has interior gaps"))
    return inner
end

# Fitted filter, centered seasonal and a note (if the filter was shortened) of the detrended
# series x, NaN where the trend is undefined
function _seasonal(model::AbstractSeasonalSmoother, x, t, p, Δt)
    N = length(x)
    inner = _finite_range(x)
    model, note = _fitting_seasonal(model, length(inner) ÷ p)
    fitted = _fit(model, x[inner], t[inner]; period = p * Δt)
    s = fill(NaN, N)
    s[inner] = predict(fitted)
    for i in 1:(first(inner) - 1)
        s[i] = s[i + cld(first(inner) - i, p) * p]
    end
    for i in (last(inner) + 1):N
        s[i] = s[i - cld(i - last(inner), p) * p]
    end
    center = predict(_fit(MovingAverage(; bandwidth = p * Δt), s, t))
    defined = _finite_range(center)
    offset = [center[clamp(i, first(defined), last(defined))] for i in 1:N]
    return fitted, s - offset, note
end

_fitting_seasonal(model::StableSeasonal, _) = model, nothing

function _fitting_seasonal(model::SeasonalMovingAverage, cycles)
    _min_cycles(model) ≤ cycles && return model, nothing
    note = "S$(model.n)×$(model.m) needs $(_min_cycles(model)) cycles, got $cycles"
    for (n, m) in ((3, 5), (3, 3))
        shorter = SeasonalMovingAverage(n, m)
        if _min_cycles(shorter) ≤ min(cycles, _min_cycles(model) - 1)
            return shorter, note
        end
    end
    return StableSeasonal(), note
end

# Global moving seasonality ratio of X-11, table D9A (Ladiray & Quenneville, 2001)
function _moving_seasonality_ratio(detrended, p)
    ΣI = ΣS = 0.0
    for phase in 1:p
        x = detrended[phase:p:end]
        n = length(x)
        extended = [fill(mean(x[1:3]), 3); x; fill(mean(x[(end - 2):end]), 3)]
        seasonal = [mean(view(extended, k:(k + 6))) for k in 1:n]
        irregular = x - seasonal
        ΣI += n * mean(abs, diff(irregular))
        ΣS += n * mean(abs, diff(seasonal))
    end
    ΣS > 0 && return ΣI / ΣS
    return ΣI > 0 ? Inf : 0.0
end

const _MSR_MIN_CYCLES = 7

# X-11's seasonal filter from the MSR, dropping the last cycle in the gray zones up to five
# times, then S3×5; returns the filter and the last MSR
function _msr_filter(detrended, p)
    msr = NaN
    for attempt in 0:5
        msr = _moving_seasonality_ratio(detrended, p)
        msr < 2.5 && return SeasonalMovingAverage(3, 3), msr
        3.5 < msr < 5.5 && return SeasonalMovingAverage(3, 5), msr
        msr > 6.5 && return SeasonalMovingAverage(3, 9), msr
        attempt < 5 && length(detrended) - p ≥ _MSR_MIN_CYCLES * p || break
        detrended = detrended[1:(end - p)]
    end
    return SeasonalMovingAverage(3, 5), msr
end
