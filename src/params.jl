function _param_wrap(v::Real, fixed::Bool)
    (v == 0.0 || !isfinite(v)) && return ParameterHandling.fixed(Float64(v))
    fixed && return ParameterHandling.fixed(Float64(v))
    return positive(Float64(v))
end

function _init_params(θ::NamedTuple, fix::AbstractVector{Symbol})
    has_seasonal = isfinite(θ.period)
    (
        σ_trend    = _param_wrap(θ.σ_trend,    :σ_trend    ∈ fix),
        l_trend    = _param_wrap(θ.l_trend,    :l_trend    ∈ fix),
        σ_seasonal = _param_wrap(has_seasonal ? θ.σ_seasonal : 0.0, :σ_seasonal ∈ fix || !has_seasonal),
        l_seasonal = _param_wrap(has_seasonal ? θ.l_seasonal : Inf,  :l_seasonal ∈ fix || !has_seasonal),
        period     = ParameterHandling.fixed(Float64(θ.period)),    # always fixed
        bandwidth  = ParameterHandling.fixed(Float64(θ.bandwidth)), # always fixed
        σ_noise    = _param_wrap(θ.σ_noise,    :σ_noise    ∈ fix),
    )
end

function _default_params()
    (σ_trend=1.0, l_trend=1.0, σ_seasonal=1.0, l_seasonal=1.0, σ_noise=1.0)
end

const _σ_KEYS = (:σ_trend, :σ_seasonal, :σ_noise)

# Merge user overrides into defaults. Defaults are in standardized space (σ = 1 means
# unit amplitude relative to the z-scored data). User-supplied σ values are in original
# data units, so they are divided by s_y before merging. l values are in time units and
# are passed through unchanged.
function _merge_params(θ_init::Union{Nothing, NamedTuple}, s_y::Real=1.0)
    base = _default_params()
    isnothing(θ_init) && return base
    θ_init_std = NamedTuple(k => (k ∈ _σ_KEYS ? Float64(v) / s_y : Float64(v))
                            for (k, v) in pairs(θ_init))
    return merge(base, θ_init_std)
end

function _destandardize_params(θ::NamedTuple, s_y::Real)
    has_seasonal = isfinite(θ.period)
    (
        σ_trend    = θ.σ_trend    * s_y,
        l_trend    = θ.l_trend,
        σ_seasonal = has_seasonal ? θ.σ_seasonal * s_y : 0.0,
        l_seasonal = has_seasonal ? θ.l_seasonal : Inf,
        period     = θ.period,
        bandwidth  = has_seasonal ? θ.bandwidth : Inf,
        σ_noise    = θ.σ_noise > 0 ? θ.σ_noise * s_y : 0.0,
    )
end
