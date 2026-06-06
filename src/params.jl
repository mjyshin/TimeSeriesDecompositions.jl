const PARAM_KEYS = (:σ_trend, :l_trend, :σ_seasonal, :l_seasonal, :period, :bandwidth, :σ_noise)

_param_wrap(v::Real, fixed::Bool) = fixed ? ParameterHandling.fixed(Float64(v)) : positive(Float64(v))

function _init_params(θ::NamedTuple, fix::AbstractVector{Symbol})
    (
        σ_trend    = _param_wrap(θ.σ_trend,    :σ_trend    ∈ fix),
        l_trend    = _param_wrap(θ.l_trend,    :l_trend    ∈ fix),
        σ_seasonal = _param_wrap(θ.σ_seasonal, :σ_seasonal ∈ fix),
        l_seasonal = _param_wrap(θ.l_seasonal, :l_seasonal ∈ fix),
        period     = _param_wrap(θ.period,     :period     ∈ fix),
        bandwidth  = _param_wrap(θ.bandwidth,  :bandwidth  ∈ fix),
        σ_noise    = _param_wrap(θ.σ_noise,    :σ_noise    ∈ fix),
    )
end

function _default_params()
    (σ_trend=1.0, l_trend=1.0, σ_seasonal=1.0, l_seasonal=1.0, period=1.0, bandwidth=Inf, σ_noise=1.0)
end

function _merge_params(init::Union{Nothing, NamedTuple})
    base = _default_params()
    isnothing(init) && return base
    return merge(base, init)
end

function _standardize_params(θ::NamedTuple, s::Real)
    (
        σ_trend    = θ.σ_trend    / s,
        l_trend    = θ.l_trend,
        σ_seasonal = θ.σ_seasonal / s,
        l_seasonal = θ.l_seasonal,
        period     = θ.period,
        bandwidth  = θ.bandwidth,
        σ_noise    = θ.σ_noise    / s,
    )
end

function _destandardize_params(θ::NamedTuple, s::Real)
    (
        σ_trend    = θ.σ_trend    * s,
        l_trend    = θ.l_trend,
        σ_seasonal = θ.σ_seasonal * s,
        l_seasonal = θ.l_seasonal,
        period     = θ.period,
        bandwidth  = θ.bandwidth,
        σ_noise    = θ.σ_noise    * s,
    )
end
