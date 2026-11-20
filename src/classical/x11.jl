"""
    X11Decomp <: AbstractDecomp

X-11-style decomposition by trend smoothers and `S(n × m)` seasonal filters, for any
integer period. The fitted smoothers are in `d.smoother`, nested by component and stage
(`d.smoother.trend.final`). Point estimates only. See `fit(X11Decomp, ...)`.
"""
struct X11Decomp <: AbstractDecomp
    t::Vector{Float64}
    trend::Component
    seasonal::Component
    noise::Component
    rate::Component
    θ::NamedTuple
    smoother::NamedTuple
    _fit_info::NamedTuple  # (Δt, grid, estimated, derived, criteria)
end

const _X11_SMOOTHERS = (
    trend = (init = MovingAverage(), interm = nothing, final = Henderson()),
    seasonal = (init = SeasonalMovingAverage(; n = 3, m = 3), final = nothing)
)

function _stages(::X11Decomp)
    return (
        (:trend, :init),
        (:seasonal, :init),
        (:trend, :interm),
        (:seasonal, :final),
        (:trend, :final)
    )
end

"""
    fit(X11Decomp, y, t=eachindex(y); period, smoother=(;)) -> X11Decomp

X-11-style decomposition of `y` at uniformly spaced times `t` with `p = period / Δt` (an
integer) samples per cycle and at least three cycles, in two passes:

 1. initial trend (`trend.init`); initial seasonal (`seasonal.init`) of the detrended
    series; intermediate trend (`trend.interm`) of the seasonally adjusted series;
 2. seasonal (`seasonal.final`) of the series detrended by the intermediate trend; trend
    (`trend.final`) of the seasonally adjusted series.

Each seasonal filter is applied where the trend is defined, extended to the ends by the
nearest estimate of the same phase, and centered by its `2 × p` moving average. The noise is
`y - trend - seasonal`. With the default smoothers, a linear trend plus a fixed seasonal
pattern is reproduced exactly.

`smoother` is a `NamedTuple` of the components `trend` (stages `init`, `interm`, `final`)
and `seasonal` (stages `init`, `final`); a component given as one smoother applies to all
its stages. The defaults are:

- `trend.init`: [`MovingAverage`](@ref)`()` over one period.
- `trend.interm`: `nothing`, the final trend smoother with minimum-revision ratio `0`.
- `trend.final`: [`Henderson`](@ref)`()` with estimated QL minimum-revision end weights.
- `seasonal.init`: [`SeasonalMovingAverage`](@ref)`(n=3, m=3)`.
- `seasonal.final`: `nothing`, `S3×3`, `S3×5` or `S3×9` by X-11's moving seasonality
  ratio (`S3×3` below seven cycles).

Trend hyperparameters left as `nothing` are derived from `p`: a [`LocalPolynomial`](@ref)
bandwidth `2hΔt` for the smallest even `2h ≥ p` (at least 4), a [`LOESS`](@ref) span of
the smallest odd number `≥ p + 3` (at least 7) of observations, and a
[`MovingAverage`](@ref) bandwidth of one period. A seasonal filter needing more cycles than
available is replaced by the longest that fits, with a note. For X-11's Musgrave end
weights on monthly data with `Δt = 1`:

```julia
fit(X11Decomp, y; period=12,
    smoother=(trend=(final=Henderson(; bandwidth=12, endpoints=musgrave(3.5)),),))
```

`d.θ` holds the period and each stage's numeric hyperparameters, named
`<quantity>_<component>_<stage>` in fitting order (e.g. `bandwidth_trend_final`);
`d._fit_info` lists those `estimated` from the data and `derived` from the period, and the
`criteria` and notes. The rate is the derivative of the not-a-knot cubic spline through the
trend. There is no extreme-value, calendar or forecast adjustment as in X-13ARIMA-SEATS.
"""
function fit(
        ::Type{X11Decomp},
        y::AbstractVector{<:Real},
        t::AbstractVector{<:Real} = eachindex(y);
        period::Real,
        smoother::NamedTuple = (;)
)
    y, t, Δt = _uniform_grid(y, t)
    p = _samples_per_period(period, Δt)
    N = length(y)
    cycles = N ÷ p
    cycles ≥ 3 || throw(ArgumentError("need at least 3 seasonal cycles, got $cycles"))
    s = _resolve_smoothers(smoother, _X11_SMOOTHERS)
    initial, derived_init = _resolve_trend(s.trend.init, p, Δt, N)
    interm = something(s.trend.interm, _interm(s.trend.final))
    interm, derived_interm = _resolve_trend(interm, p, Δt, N)
    final, derived_final = _resolve_trend(s.trend.final, p, Δt, N)
    r = _Report(period)

    # Pass 1
    trend_init = _fit(initial, y, t)
    _record!(r, trend_init, :_trend_init; derived = keys(derived_init))
    seasonal_init, seasonal_values, note = _seasonal(
        s.seasonal.init, y - predict(trend_init), t, p, Δt
    )
    _record!(r, seasonal_init, :_seasonal_init)
    isnothing(note) || push!(r.criteria, :note_seasonal_init => note)
    trend_interm = _fit(interm, y - seasonal_values, t)
    _record!(r, trend_interm, :_trend_interm; derived = keys(derived_interm))

    # Pass 2
    detrended = y - predict(trend_interm)
    inner = _finite_range(detrended)
    seasonal_model, derived, chosen = s.seasonal.final, (), ()
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
    seasonal_final, seasonal_values, note = _seasonal(
        seasonal_model, detrended, t, p, Δt
    )
    _record!(r, seasonal_final, :_seasonal_final; derived, chosen)
    isnothing(note) || push!(r.criteria, :note_seasonal_final => note)
    adjusted = y - seasonal_values
    trend_final = _fit(final, adjusted, t)
    _record!(r, trend_final, :_trend_final; derived = keys(derived_final))
    trend_values = predict(trend_final)

    noise = adjusted - trend_values
    rate = _trend_rate(trend_values, Δt)
    grid = (; t, trend = trend_values, seasonal = seasonal_values, noise, rate)
    θ, fit_info = _finish(r, X11Decomp, Δt, grid)
    fitted = (
        trend = (init = trend_init, interm = trend_interm, final = trend_final),
        seasonal = (init = seasonal_init, final = seasonal_final)
    )
    return X11Decomp(
        t,
        Component(trend_values),
        Component(seasonal_values),
        Component(noise),
        Component(rate),
        θ,
        fitted,
        fit_info
    )
end

# Trend smoother with hyperparameters left as nothing derived from p, and those values
function _resolve_trend(m::LocalPolynomial, p, Δt, _)
    isnothing(m.bandwidth) || return m, (;)
    bandwidth = max(4, 2cld(p, 2)) * Δt
    return LocalPolynomial(bandwidth, m.degree, m.kernel, m.endpoints), (; bandwidth)
end

function _resolve_trend(m::LOESS, p, _, N)
    isnothing(m.span) && isnothing(m.bandwidth) || return m, (;)
    q = max(7, _nextodd(p + 3))
    span = q / N
    return LOESS(span, nothing, m.degree), (; span, neighbors = q)
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
