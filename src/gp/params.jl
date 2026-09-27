function _param_wrap(v::Real, fixed::Bool)
    (v == 0.0 || !isfinite(v)) && return ParameterHandling.fixed(Float64(v))
    fixed && return ParameterHandling.fixed(Float64(v))
    return positive(Float64(v))
end

# Lower bound on l_trend that keeps the lengthscales ordered,
#     l_trend > max(bandwidth, period) ≥ bandwidth ≥ period > seasonal timescale > 0
# where 0 is the (white) noise.
# A periodic kernel's correlation range is at most period/2, and the zero-mean seasonal
# kernel cannot mimic a trend, so l_trend is the only parameter bounded for the ordering.
# With bandwidth = Inf (static seasonality) the bound is the period; without seasonality,
# there is no bound.
function _trend_lower_bound(period::Real, bandwidth::Real)
    isfinite(period) || return 0.0
    return isfinite(bandwidth) ? Float64(max(bandwidth, period)) : Float64(period)
end

# l_trend = lb + exp(x): strictly above the bound, unconstrained for the optimizer
function _trend_wrap(v::Real, fixed::Bool, lb::Real)
    lb > 0 || return _param_wrap(v, fixed)
    if fixed || !isfinite(v)
        v > lb || throw(
            ArgumentError(
                "fixed l_trend = $v must exceed max(bandwidth, period) = $lb so that " *
                "the trend is smoother than the seasonal component",
            ),
        )
        return ParameterHandling.fixed(Float64(v))
    end
    return positive(Float64(v), exp, lb)
end

# Upper bound on l_seasonal (on the unit circle, where chord distances are ≤ 2). Beyond
# it, any smooth periodic kernel is already ≈ cos(2πΔ/period), the likelihood is flat and
# the zero-mean kernel loses positive semi-definiteness to cancellation (1 - a₀ ~ 1/l²).
const _L_SEASONAL_MAX = 10.0

function _seasonal_wrap(v::Real, fixed::Bool)
    0 < v < Inf || throw(
        ArgumentError(
            "l_seasonal = $v must be positive and finite; use period=Inf (or omit it) " *
            "for no seasonal component",
        ),
    )
    if fixed
        v <= _L_SEASONAL_MAX ||
            throw(ArgumentError("fixed l_seasonal = $v must be at most $_L_SEASONAL_MAX"))
        return ParameterHandling.fixed(Float64(v))
    end
    v < _L_SEASONAL_MAX || @info "θ_init.l_seasonal = $v is not below " *
        "$_L_SEASONAL_MAX; starting the optimizer from l_seasonal = 1"
    # bounded needs a starting value strictly inside (0, _L_SEASONAL_MAX)
    v₀ = v < _L_SEASONAL_MAX ? clamp(Float64(v), 1e-6, _L_SEASONAL_MAX - 1e-6) : 1.0
    return bounded(v₀, 0.0, _L_SEASONAL_MAX)
end

# Initial l_trend: the default starts at 2lb; a user value on or below the bound is moved
function _initial_l_trend(v::Real, user_given::Bool, fixed::Bool, lb::Real)
    (lb > 0 && !fixed && v <= lb) || return Float64(v)
    user_given && @info "θ_init.l_trend = $v is not above max(bandwidth, period) = " *
        "$lb; starting the optimizer from l_trend = $(2lb)"
    return 2lb
end

function _init_params(θ::NamedTuple, fix::AbstractVector{Symbol})
    has_seasonal = isfinite(θ.period)
    return (
        σ_trend=_param_wrap(θ.σ_trend, :σ_trend ∈ fix),
        l_trend=_trend_wrap(
            θ.l_trend, :l_trend ∈ fix, _trend_lower_bound(θ.period, θ.bandwidth)
        ),
        σ_seasonal=_param_wrap(
            has_seasonal ? θ.σ_seasonal : 0.0, :σ_seasonal ∈ fix || !has_seasonal
        ),
        l_seasonal=if has_seasonal
            _seasonal_wrap(θ.l_seasonal, :l_seasonal ∈ fix)
        else
            ParameterHandling.fixed(Inf)
        end,
        period=ParameterHandling.fixed(Float64(θ.period)),    # always fixed
        bandwidth=ParameterHandling.fixed(Float64(θ.bandwidth)), # always fixed
        σ_noise=_param_wrap(θ.σ_noise, :σ_noise ∈ fix),
    )
end

function _default_params()
    return (σ_trend=1.0, l_trend=1.0, σ_seasonal=1.0, l_seasonal=1.0, σ_noise=1.0)
end

const _σ_KEYS = (:σ_trend, :σ_seasonal, :σ_noise)

# Merge user overrides into defaults. Defaults are in standardized space (σ = 1 means
# unit amplitude relative to the z-scored data). User-supplied σ values are in original
# data units, so they are divided by s_y before merging. l values are in time units and
# are passed through unchanged.
function _merge_params(θ_init::Union{Nothing,NamedTuple}, s_y::Real=1.0)
    base = _default_params()
    isnothing(θ_init) && return base
    θ_init_std = NamedTuple(
        k => (k ∈ _σ_KEYS ? Float64(v) / s_y : Float64(v)) for (k, v) in pairs(θ_init)
    )
    return merge(base, θ_init_std)
end

function _destandardize_params(θ::NamedTuple, s_y::Real)
    has_seasonal = isfinite(θ.period)
    return (
        σ_trend=θ.σ_trend * s_y,
        l_trend=θ.l_trend,
        σ_seasonal=has_seasonal ? θ.σ_seasonal * s_y : 0.0,
        l_seasonal=has_seasonal ? θ.l_seasonal : Inf,
        period=θ.period,
        bandwidth=has_seasonal ? θ.bandwidth : Inf,
        σ_noise=θ.σ_noise > 0 ? θ.σ_noise * s_y : 0.0,
    )
end
