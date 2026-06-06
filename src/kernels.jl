function _seasonal_kernel(l_seasonal::Real, period::Real, bandwidth::Real)
    k_per = with_lengthscale(SEKernel(), l_seasonal) ∘ PeriodicTransform(1 / period)
    isfinite(bandwidth) || return k_per
    return k_per * with_lengthscale(SEKernel(), bandwidth)
end

function _build_gp(θ::NamedTuple; rate::Bool=false)
    k_trend    = θ.σ_trend^2    * with_lengthscale(SEKernel(), θ.l_trend)
    k_seasonal = θ.σ_seasonal^2 * _seasonal_kernel(θ.l_seasonal, θ.period, θ.bandwidth)
    k_noise    = θ.σ_noise^2    * WhiteKernel()

    if rate
        return @gppp let
            x̄ = GP(k_trend)
            x̃ = GP(k_seasonal)
            ϵ = GP(k_noise)
            y = x̄ + x̃ + ϵ
            r = derivative(x̄)
        end
    end

    return @gppp let
        x̄ = GP(k_trend)
        x̃ = GP(k_seasonal)
        ϵ = GP(k_noise)
        y = x̄ + x̃ + ϵ
    end
end
