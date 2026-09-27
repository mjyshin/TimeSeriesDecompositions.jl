const _default_kernels = (trend=SEKernel(), seasonal=SEKernel(), bandwidth=SEKernel())

function _merge_kernels(k::Union{Nothing,NamedTuple})
    isnothing(k) && return _default_kernels
    valid = keys(_default_kernels)
    for key in keys(k)
        key ∈ valid || throw(ArgumentError("Unknown kernel key :$key. Valid keys: $valid"))
    end
    return merge(_default_kernels, k)
end

function _seasonal_kernel(
    l_seasonal::Real, period::Real, bandwidth::Real, k_seasonal, k_bandwidth
)
    k_per = with_lengthscale(k_seasonal, l_seasonal) ∘ PeriodicTransform(1 / period)
    isfinite(bandwidth) || return k_per
    return k_per * with_lengthscale(k_bandwidth, bandwidth)
end

function _build_gp(θ::NamedTuple, kernels::NamedTuple=_default_kernels)
    k_trend = θ.σ_trend^2 * with_lengthscale(kernels.trend, θ.l_trend)
    has_seasonal = isfinite(θ.period)
    has_noise = θ.σ_noise > 0

    if has_seasonal && has_noise
        k_seasonal =
            θ.σ_seasonal^2 * _seasonal_kernel(
                θ.l_seasonal, θ.period, θ.bandwidth, kernels.seasonal, kernels.bandwidth
            )
        k_noise = θ.σ_noise^2 * WhiteKernel()
        return @gppp let
            x̄ = GP(k_trend)
            x̃ = GP(k_seasonal)
            ϵ = GP(k_noise)
            y = x̄ + x̃ + ϵ
        end
    elseif has_seasonal
        k_seasonal =
            θ.σ_seasonal^2 * _seasonal_kernel(
                θ.l_seasonal, θ.period, θ.bandwidth, kernels.seasonal, kernels.bandwidth
            )
        return @gppp let
            x̄ = GP(k_trend)
            x̃ = GP(k_seasonal)
            y = x̄ + x̃
        end
    elseif has_noise
        k_noise = θ.σ_noise^2 * WhiteKernel()
        return @gppp let
            x̄ = GP(k_trend)
            ϵ = GP(k_noise)
            y = x̄ + ϵ
        end
    else
        return @gppp let
            x̄ = GP(k_trend)
            y = x̄
        end
    end
end
