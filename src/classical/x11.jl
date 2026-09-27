"""
    X11Decomp <: AbstractDecomp

X-11-style seasonal-trend decomposition from classical smoothers: a trend smoother
(Henderson by default, or any [`AbstractTrendSmoother`](@ref)) and `S(n × m)` seasonal
filters, for any integer period. The fitted smoothers are in `d.θ.smoothers` and the
hyperparameters chosen from the data in `d.θ.estimated`; `d.θ.grid` keeps the components
at the observed times, from which `predict(d, t_new)` interpolates. Point estimates only
(`std` is `NaN`). See `fit(X11Decomp, ...)`.
"""
struct X11Decomp <: AbstractDecomp
    t::Vector{Float64}
    trend::Component
    seasonal::Component
    noise::Component
    rate::Component
    θ::NamedTuple
end

"""
    fit(X11Decomp, y, t=eachindex(y); period, kwargs...) -> X11Decomp

X-11-style decomposition of `y` observed at uniformly spaced times `t`, with `p` samples
per seasonal cycle, in two passes:

 1. a centred `2 × p` (even `p`) or `p`-term (odd `p`) moving average gives an initial
    trend, undefined in the first and last `p/2` points; a seasonal filter of the
    detrended series gives a preliminary seasonal; a trend smoother of the seasonally
    adjusted series gives a preliminary trend;
 2. a seasonal filter of the series detrended by the preliminary trend gives the
    seasonal; a trend smoother of the seasonally adjusted series gives the trend.

In each pass the seasonal filter is applied to every cycle-subseries where the trend is
defined, extended to the other points by repeating the nearest estimate of the same
phase (which holds a changing seasonal amplitude fixed over the ends), and centred by
subtracting its `2 × p` moving average. The noise is `y - trend - seasonal`. With the
default smoothers, a linear trend plus a fixed seasonal pattern is reproduced exactly,
including at the ends.

# Keyword arguments
- `period`: seasonal period in units of `t` (required); `p = period / Δt` must be an
  integer, and the series must span at least three cycles.
- `trend`: an [`AbstractTrendSmoother`](@ref) for the final trend (default
  [`Henderson`](@ref)`()`, with QL minimum-revision end weights whose ratio is estimated
  from the seasonally adjusted series). Hyperparameters left as `nothing` are derived
  from `p`: a [`LocalPolynomial`](@ref) window is the smallest odd integer `≥ p + 1` (at
  least 5), giving the 5- and 13-term Henderson filters of X-11 for quarterly and
  monthly data; a [`LOESS`](@ref) span covers the smallest odd number `≥ p + 3` (at
  least 7) of observations, which matches that Henderson filter's cutoff and noise
  reduction at degree 2; a [`MovingAverage`](@ref) width is the period. Use
  `Henderson(; window=13, endpoints=musgrave(3.5))` for X-11's Musgrave weights.
- `preliminary_trend`: the first-pass trend smoother; `nothing` (default) uses `trend`
  with the same hyperparameters and minimum-revision end weights of ratio `0`.
- `preliminary_seasonal`: the first-pass seasonal filter (default
  [`SeasonalMovingAverage`](@ref)`(n=3, m=3)`).
- `seasonal`: the final seasonal filter; `nothing` (default) chooses among `S3×3`,
  `S3×5` and `S3×9` by X-11's global moving seasonality ratio (MSR; `S3×3` below seven
  cycles), whose thresholds, in units of cycles, were calibrated on monthly and
  quarterly data. [`StableSeasonal`](@ref)`()` gives a fixed pattern.

A seasonal filter needing more cycles than available (`n + m - 1`) is replaced by the
longest one that fits, and by a stable seasonal below five cycles. Hyperparameters chosen
from the data are reported in one `@info` message and stored in `d.θ.estimated`. Unlike
X-13ARIMA-SEATS, there is no extreme-value adjustment, forecast extension or calendar
adjustment.

The rate is the derivative, per unit of `t`, of the not-a-knot cubic spline through the
trend.
"""
function fit(
    ::Type{X11Decomp},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Real,
    trend::AbstractTrendSmoother=Henderson(),
    preliminary_trend::Union{Nothing,AbstractTrendSmoother}=nothing,
    preliminary_seasonal::AbstractSeasonalSmoother=SeasonalMovingAverage(; n=3, m=3),
    seasonal::Union{Nothing,AbstractSeasonalSmoother}=nothing,
)
    y, t, Δt = _uniform_grid(y, t)
    p = _samples_per_period(period, Δt)
    N = length(y)
    cycles = N ÷ p
    cycles ≥ 3 || throw(ArgumentError("need at least 3 seasonal cycles, got $cycles"))
    trend, estimated = _resolve_trend(trend, p, Δt, N)
    preliminary = something(preliminary_trend, _preliminary(trend))
    preliminary, derived = _resolve_trend(preliminary, p, Δt, N)
    isnothing(preliminary_trend) ||
        (estimated = (; estimated..., _prefix_keys(:preliminary_, derived)...))

    # Pass 1
    initial_trend = _fit(MovingAverage(; width=p * Δt), y, t)
    preliminary_seasonal_fit, seasonal_values, note = _seasonal(
        preliminary_seasonal, y - predict(initial_trend), t, p, Δt
    )
    isnothing(note) || (estimated = (; estimated..., preliminary_seasonal_note=note))
    preliminary_trend_fit = _fit(preliminary, y - seasonal_values, t)
    preliminary_estimated = _prefix_keys(:preliminary_, preliminary_trend_fit.estimated)
    estimated = (; estimated..., preliminary_estimated...)

    # Pass 2
    detrended = y - predict(preliminary_trend_fit)
    inner = _finite_range(detrended)
    if isnothing(seasonal)
        available = length(inner) ÷ p
        if available < _MSR_MIN_CYCLES
            msr_note = "MSR needs $_MSR_MIN_CYCLES cycles, got $available; using S3×3"
            seasonal = SeasonalMovingAverage(; n=3, m=3)
            estimated = (; estimated..., msr_note)
        else
            seasonal, msr = _msr_filter(detrended[inner], p)
            estimated = (; estimated..., msr)
        end
        estimated = (; estimated..., seasonal)
    end
    seasonal_fit, seasonal_values, note = _seasonal(seasonal, detrended, t, p, Δt)
    if !isnothing(note)
        estimated = (; estimated..., seasonal=seasonal_fit.model, seasonal_note=note)
    end
    adjusted = y - seasonal_values
    trend_fit = _fit(trend, adjusted, t)
    trend_values = predict(trend_fit)
    estimated = (; estimated..., trend_fit.estimated...)

    _log_estimated(X11Decomp, estimated)
    smoothers = (;
        initial_trend,
        preliminary_seasonal=preliminary_seasonal_fit,
        preliminary_trend=preliminary_trend_fit,
        seasonal=seasonal_fit,
        trend=trend_fit,
    )
    noise = adjusted - trend_values
    rate = _trend_rate(trend_values, Δt)
    grid = (; t, trend=trend_values, seasonal=seasonal_values, noise, rate)
    return X11Decomp(
        t,
        Component(trend_values),
        Component(seasonal_values),
        Component(noise),
        Component(rate),
        (; period=Float64(period), Δt, smoothers, estimated, grid),
    )
end

function _prefix_keys(prefix::Symbol, nt::NamedTuple)
    return NamedTuple(Symbol(prefix, k) => v for (k, v) in pairs(nt))
end

# Replace the hyperparameters of a trend smoother left as nothing by those derived from
# the p samples per period; returns the smoother and the derived values
function _resolve_trend(m::LocalPolynomial, p, _, _)
    isnothing(m.window) || return m, (;)
    window = max(5, _nextodd(p + 1))
    return LocalPolynomial(window, m.degree, m.kernel, m.endpoints), (; trend_window=window)
end

function _resolve_trend(m::LOESS, p, _, N)
    isnothing(m.span) && isnothing(m.bandwidth) || return m, (;)
    q = max(7, _nextodd(p + 3))
    span = q / N
    return LOESS(span, nothing, m.degree), (; trend_span=span, trend_neighbours=q)
end

function _resolve_trend(m::MovingAverage, p, Δt, _)
    isnothing(m.width) || return m, (;)
    width = p * Δt
    return MovingAverage(width, m.endpoints, m.gapfactor), (; trend_width=width)
end

# The preliminary trend: the final smoother with minimum-revision end weights of ratio 0
_preliminary(m::AbstractTrendSmoother) = m

function _preliminary(m::LocalPolynomial)
    return LocalPolynomial(m.window, m.degree, m.kernel, _preliminary(m.endpoints))
end

function _preliminary(m::MovingAverage)
    return MovingAverage(m.width, _preliminary(m.endpoints), m.gapfactor)
end

_preliminary(e::AbstractEndpoints) = e
_preliminary(e::MinimumRevision) = MinimumRevision(e.degree, 0.0)

# Indices of the finite values of x, which must be contiguous
function _finite_range(x)
    idx = findall(isfinite, x)
    isempty(idx) && throw(ArgumentError("the trend is undefined everywhere"))
    inner = first(idx):last(idx)
    length(idx) == length(inner) || throw(ArgumentError("the trend has interior gaps"))
    return inner
end

# Seasonal component of the detrended series x (NaN where the trend is undefined):
# filter each cycle-subseries where x is defined, repeat the nearest estimate of the same
# phase elsewhere, and centre with a 2 × p moving average (its ends extended as
# constants). A filter needing more cycles than available is replaced by the longest that
# fits; returns the fitted filter, the seasonal and a note if the filter was replaced.
function _seasonal(model::AbstractSeasonalSmoother, x, t, p, Δt)
    N = length(x)
    inner = _finite_range(x)
    model, note = _fitting_seasonal(model, length(inner) ÷ p)
    fitted = _fit(model, x[inner], t[inner]; period=p * Δt)
    s = fill(NaN, N)
    s[inner] = predict(fitted)
    for i in 1:(first(inner) - 1)
        s[i] = s[i + cld(first(inner) - i, p) * p]
    end
    for i in (last(inner) + 1):N
        s[i] = s[i - cld(i - last(inner), p) * p]
    end
    centre = predict(_fit(MovingAverage(; width=p * Δt), s, t))
    defined = _finite_range(centre)
    offset = [centre[clamp(i, first(defined), last(defined))] for i in 1:N]
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

# Global moving seasonality ratio of X-11 (table D9A; Ladiray & Quenneville, 2001): in
# each cycle-subseries, the seasonal S is a 7-term moving average of the detrended values
# (extended at each end by three copies of the mean of the first or last three) and the
# irregular is I = detrended - S; the ratio is Σⱼ Nⱼ Īⱼ / Σⱼ Nⱼ S̄ⱼ, where Īⱼ and S̄ⱼ are the
# mean absolute cycle-to-cycle changes of I and S at phase j and Nⱼ its number of cycles
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

# X-11's choice of seasonal filter from the MSR: S3×3 below 2.5, S3×5 in (3.5, 5.5),
# S3×9 above 6.5. The closed grey zones [2.5, 3.5] and [5.5, 6.5] trigger a recomputation
# without the last cycle, up to five times, after which S3×5 is used. Returns the filter
# and the last MSR computed.
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
