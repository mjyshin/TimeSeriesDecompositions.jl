# Starting values on or within rounding of a bound, e.g. warm starts from a fit whose
# optimum saturated there, are moved just inside it, where the transforms are defined
_inside(v::Real, lb::Real) = max(Float64(v), lb + 1e-6 * max(abs(lb), 1e-6))

function _param_wrap(v::Real, fixed::Bool)
    (v == 0.0 || !isfinite(v)) && return ParameterHandling.fixed(Float64(v))
    fixed && return ParameterHandling.fixed(Float64(v))
    ε = sqrt(eps(Float64))    # the lower bound of `positive`
    return positive(_inside(v, ε), exp, ε)
end

# Lower bound on l_trend that separates the trend from the seasonal component,
#     l_trend > period > seasonal timescale > 0 (noise), bandwidth ≥ period.
# A periodic kernel's correlation range is at most period/2, and the zero-mean seasonal
# kernel cannot mimic a trend: with the envelope, its spectrum is Σ_{k≠0} aₖ S_env(f - k/P),
# whose leakage to f = 0 is S_env(1/P)/S_env(0) = (1 + 4π²(B/P)²/5)⁻³ ≤ 1.4e-3 for a
# Matérn-5/2 envelope of bandwidth B ≥ P. The trend competes with the seasonal component
# only for l_trend ~ period, so l_trend need not exceed the bandwidth, which then stays
# free to be estimated. Without seasonality there is no bound.
_trend_lower_bound(period::Real) = isfinite(period) ? Float64(period) : 0.0

# l_trend = lb + exp(x): strictly above the bound, unconstrained for the optimizer
function _trend_wrap(v::Real, fixed::Bool, lb::Real)
    lb > 0 || return _param_wrap(v, fixed)
    if fixed || !isfinite(v)
        v > lb || throw(
            ArgumentError(
                "fixed l_trend = $v must exceed the period $lb so that the trend is " *
                "smoother than the seasonal component",
            ),
        )
        return ParameterHandling.fixed(Float64(v))
    end
    return positive(_inside(v, lb), exp, lb)
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
    v ≤ _L_SEASONAL_MAX || @info "θ_init.l_seasonal = $v is above " *
        "$_L_SEASONAL_MAX; starting the optimizer from l_seasonal = 1"
    # bounded needs a starting value strictly inside (0, _L_SEASONAL_MAX); one at the cap
    # (e.g. a warm start from a saturated fit) is moved just below it
    v₀ = v ≤ _L_SEASONAL_MAX ? clamp(Float64(v), 1e-6, _L_SEASONAL_MAX - 1e-6) : 1.0
    return bounded(v₀, 0.0, _L_SEASONAL_MAX)
end

# Initial l_trend: the default starts at 2lb; a user value on or below the bound is moved
function _initial_l_trend(v::Real, user_given::Bool, fixed::Bool, lb::Real)
    (lb > 0 && !fixed && v <= lb) || return Float64(v)
    user_given && @info "θ_init.l_trend = $v is not above the period $lb; starting " *
        "the optimizer from l_trend = $(2lb)"
    return 2lb
end

# The envelope enters as ρ = period / bandwidth ∈ [0, 1], so that the static model
# (bandwidth = Inf) is the finite boundary ρ = 0. The NLL is even in ρ there (the
# envelope is 1 - c(ρΔ/P)² + … for smooth kernels), so the optimizer cannot reach ρ = 0
# from inside; the static model is fitted separately with ρ fixed at 0.
function _bandwidth_wrap(ρ::Real, fixed::Bool)
    fixed && return ParameterHandling.fixed(Float64(ρ))
    return bounded(clamp(Float64(ρ), 1e-4, 1 - 1e-3), 0.0, 1.0)
end

function _init_params(θ::NamedTuple, fix::AbstractVector{Symbol})
    has_seasonal = isfinite(θ.period)
    return (
        σ_trend=_param_wrap(θ.σ_trend, :σ_trend ∈ fix),
        l_trend=_trend_wrap(θ.l_trend, :l_trend ∈ fix, _trend_lower_bound(θ.period)),
        σ_seasonal=_param_wrap(
            has_seasonal ? θ.σ_seasonal : 0.0, :σ_seasonal ∈ fix || !has_seasonal
        ),
        l_seasonal=if has_seasonal
            _seasonal_wrap(θ.l_seasonal, :l_seasonal ∈ fix)
        else
            ParameterHandling.fixed(Inf)
        end,
        period=ParameterHandling.fixed(Float64(θ.period)),    # always fixed
        ρ=if has_seasonal
            _bandwidth_wrap(θ.ρ, :bandwidth ∈ fix)
        else
            ParameterHandling.fixed(0.0)
        end,
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
        bandwidth=has_seasonal && θ.ρ > 0 ? θ.period / θ.ρ : Inf,
        σ_noise=θ.σ_noise > 0 ? θ.σ_noise * s_y : 0.0,
    )
end
