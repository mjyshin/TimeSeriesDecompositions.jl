# A free hyperparameter with a starting value in (lb, ub): lb + exp(x) when ub = Inf, else
# lb + (ub - lb) logistic(x); fixed hyperparameters are plain numbers
struct _Free
    init::Float64
    lb::Float64
    ub::Float64
end

_value(p::_Free, x) = isinf(p.ub) ? p.lb + exp(x) : p.lb + (p.ub - p.lb) / (1 + exp(-x))
function _unconstrained(p::_Free)
    isinf(p.ub) && return log(p.init - p.lb)
    u = (p.init - p.lb) / (p.ub - p.lb)
    return log(u / (1 - u))
end

# Free hyperparameters, their unconstrained starting point and the map back to θ
function _flatten(spec::NamedTuple)
    free = [k for (k, v) in pairs(spec) if v isa _Free]
    x₀ = Float64[_unconstrained(spec[k]) for k in free]
    function unflatten(x::AbstractVector{T}) where {T}
        i = 0
        return map(spec) do p
            p isa _Free ? _value(p, x[i += 1]) : convert(T, p)
        end
    end
    return x₀, unflatten, free
end

# Move a starting value on (or within rounding of) a lower bound just inside it
_inside(v::Real, lb::Real) = max(Float64(v), lb + 1e-6 * max(abs(lb), 1e-6))

function _param_wrap(v::Real, isfixed::Bool)
    (isfixed || v == 0 || !isfinite(v)) && return Float64(v)
    ϵ = sqrt(eps(Float64))
    return _Free(_inside(v, ϵ), ϵ, Inf)
end

# l_trend > period separates the trend from the zero-mean seasonal component; no bound
# without seasonality
_trend_lower_bound(period::Real) = isfinite(period) ? Float64(period) : 0.0

function _trend_wrap(v::Real, isfixed::Bool, lb::Real)
    lb > 0 || return _param_wrap(v, isfixed)
    if isfixed || !isfinite(v)
        v > lb || throw(ArgumentError("fixed l_trend = $v must exceed the period $lb"))
        return Float64(v)
    end
    return _Free(_inside(v, lb), lb, Inf)
end

# Beyond this cap (chord distances on the unit circle are ≤ 2) every smooth periodic kernel
# is ≈ cos(2πΔ/period) and the zero-mean kernel loses definiteness to cancellation
const _L_SEASONAL_MAX = 10.0

function _seasonal_wrap(v::Real, isfixed::Bool)
    0 < v < Inf || throw(ArgumentError("l_seasonal = $v must be positive and finite; use " *
                        "period=Inf for no seasonal component"))
    if isfixed
        v <= _L_SEASONAL_MAX ||
            throw(ArgumentError("fixed l_seasonal = $v must be at most $_L_SEASONAL_MAX"))
        return Float64(v)
    end
    v ≤ _L_SEASONAL_MAX ||
        @info "θ_init.l_seasonal = $v is above $_L_SEASONAL_MAX; starting from l_seasonal = 1"
    v₀ = v ≤ _L_SEASONAL_MAX ? clamp(Float64(v), 1e-6, _L_SEASONAL_MAX - 1e-6) : 1.0
    return _Free(v₀, 0.0, _L_SEASONAL_MAX)
end

function _initial_l_trend(v::Real, user_given::Bool, isfixed::Bool, lb::Real)
    (lb > 0 && !isfixed && v <= lb) || return Float64(v)
    user_given &&
        @info "θ_init.l_trend = $v is not above the period $lb; starting from l_trend = $(2lb)"
    return 2lb
end

# ρ = period / bandwidth ∈ [0, 1]; the NLL is even in ρ at the static boundary ρ = 0, which
# the optimizer cannot reach, so the static model is fitted separately
function _bandwidth_wrap(ρ::Real, isfixed::Bool)
    isfixed && return Float64(ρ)
    return _Free(clamp(Float64(ρ), 1e-4, 1 - 1e-3), 0.0, 1.0)
end

function _init_params(θ::NamedTuple, fix::AbstractVector{Symbol})
    has_seasonal = isfinite(θ.period)
    return (
        σ_trend = _param_wrap(θ.σ_trend, :σ_trend ∈ fix),
        l_trend = _trend_wrap(θ.l_trend, :l_trend ∈ fix, _trend_lower_bound(θ.period)),
        σ_seasonal = _param_wrap(has_seasonal ? θ.σ_seasonal : 0.0,
            :σ_seasonal ∈ fix || !has_seasonal),
        l_seasonal = has_seasonal ? _seasonal_wrap(θ.l_seasonal, :l_seasonal ∈ fix) : Inf,
        period = Float64(θ.period),
        ρ = has_seasonal ? _bandwidth_wrap(θ.ρ, :bandwidth ∈ fix) : 0.0,
        σ_noise = _param_wrap(θ.σ_noise, :σ_noise ∈ fix)
    )
end

const _σ_KEYS = (:σ_trend, :σ_seasonal, :σ_noise)

# Starting values in standardized units: defaults of 1, overridden by θ_init (whose σ are in
# data units)
function _merge_params(θ_init::NamedTuple, s_y::Real)
    base = (σ_trend = 1.0, l_trend = 1.0, σ_seasonal = 1.0, l_seasonal = 1.0, σ_noise = 1.0)
    user = NamedTuple(k => (k ∈ _σ_KEYS ? Float64(v) / s_y : Float64(v))
    for (k, v) in pairs(θ_init))
    return merge(base, user)
end

function _destandardize_params(θ::NamedTuple, s_y::Real)
    has_seasonal = isfinite(θ.period)
    return (
        σ_trend = θ.σ_trend * s_y,
        l_trend = θ.l_trend,
        σ_seasonal = has_seasonal ? θ.σ_seasonal * s_y : 0.0,
        l_seasonal = has_seasonal ? θ.l_seasonal : Inf,
        bandwidth = has_seasonal && θ.ρ > 0 ? θ.period / θ.ρ : Inf,
        σ_noise = θ.σ_noise > 0 ? θ.σ_noise * s_y : 0.0
    )
end
