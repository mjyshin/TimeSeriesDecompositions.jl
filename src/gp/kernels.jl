const _DEFAULT_KERNEL = Matern52Kernel()

# Resolve the `kernel` kwarg to (trend, seasonal=(effect, envelope)). A kernel at any level
# applies to every component below it; unspecified components take the default.
_resolve_kernels(k::Kernel) = (trend=k, seasonal=(effect=k, envelope=k))
_resolve_kernels(k::NamedTuple) = _resolve_kernels(_DEFAULT_KERNEL, k)
function _resolve_kernels(default::Kernel, k::NamedTuple)
    _check_kernel_keys(k, (:trend, :seasonal), "kernel")
    trend = get(k, :trend, default)
    trend isa Kernel || throw(ArgumentError("kernel.trend must be a Kernel"))
    return (trend=trend, seasonal=_resolve_seasonal(get(k, :seasonal, default), default))
end
function _resolve_kernels(k)
    throw(ArgumentError("kernel must be a Kernel or a NamedTuple, got $(typeof(k))"))
end

_resolve_seasonal(k::Kernel, _) = (effect=k, envelope=k)
function _resolve_seasonal(k::NamedTuple, default::Kernel)
    _check_kernel_keys(k, (:effect, :envelope), "kernel.seasonal")
    effect, envelope = get(k, :effect, default), get(k, :envelope, default)
    (effect isa Kernel && envelope isa Kernel) || throw(
        ArgumentError("kernel.seasonal.effect and kernel.seasonal.envelope must be Kernels")
    )
    return (effect=effect, envelope=envelope)
end
function _resolve_seasonal(k, _)
    throw(
        ArgumentError("kernel.seasonal must be a Kernel or a NamedTuple, got $(typeof(k))")
    )
end

function _check_kernel_keys(k::NamedTuple, valid, name)
    for key in keys(k)
        key ∈ valid || throw(ArgumentError("Unknown $name key :$key. Valid keys: $valid"))
    end
end

const _default_kernels = _resolve_kernels(_DEFAULT_KERNEL)

# Periodic kernel with its constant Fourier coefficient a₀ removed and rescaled to unit
# variance, k̃ = (k - a₀) / (1 - a₀). Without this, the envelope turns σ²a₀ of the
# seasonal variance into a second smooth trend GP of lengthscale `bandwidth`, which
# competes with the trend. Dropping a non-negative Fourier coefficient keeps k̃ positive
# semi-definite.
struct _ZeroMeanKernel{K<:Kernel,T<:Real} <: Kernel
    kernel::K
    a₀::T
end

(k::_ZeroMeanKernel)(x, y) = (k.kernel(x, y) - k.a₀) / (1 - k.a₀)

_zero_mean(k::_ZeroMeanKernel, K) = (K .- k.a₀) ./ (1 - k.a₀)

function KernelFunctions.kernelmatrix(k::_ZeroMeanKernel, x::AbstractVector)
    return _zero_mean(k, kernelmatrix(k.kernel, x))
end
function KernelFunctions.kernelmatrix(
    k::_ZeroMeanKernel, x::AbstractVector, y::AbstractVector
)
    return _zero_mean(k, kernelmatrix(k.kernel, x, y))
end
function KernelFunctions.kernelmatrix_diag(k::_ZeroMeanKernel, x::AbstractVector)
    return _zero_mean(k, kernelmatrix_diag(k.kernel, x))
end
function KernelFunctions.kernelmatrix_diag(
    k::_ZeroMeanKernel, x::AbstractVector, y::AbstractVector
)
    return _zero_mean(k, kernelmatrix_diag(k.kernel, x, y))
end

# a₀ is the mean of k(0, u) over one period, by the trapezoid rule, which works for any
# base kernel. Aliasing makes it overestimate a₀ (breaking positive semi-definiteness),
# spectrally little for smooth kernels but O(1/n²) for Matérn-1/2, hence the large n.
function _zero_mean_periodic(k_per::Kernel, period::Real; n::Int=1024)
    u = collect(range(0.0, period; length=n + 1)[1:n])
    a₀ = mean(kernelmatrix(k_per, [0.0], u))
    return _ZeroMeanKernel(k_per, a₀)
end

function _seasonal_kernel(
    l_seasonal::Real, period::Real, bandwidth::Real, kernels::NamedTuple
)
    k_per = with_lengthscale(kernels.effect, l_seasonal) ∘ PeriodicTransform(1 / period)
    k_per = _zero_mean_periodic(k_per, period)
    isfinite(bandwidth) || return k_per
    return k_per * with_lengthscale(kernels.envelope, bandwidth)
end

function _build_gp(θ::NamedTuple, kernels::NamedTuple=_default_kernels)
    k_trend = θ.σ_trend^2 * with_lengthscale(kernels.trend, θ.l_trend)
    has_seasonal = isfinite(θ.period)
    has_noise = θ.σ_noise > 0

    if has_seasonal && has_noise
        k_seasonal =
            θ.σ_seasonal^2 *
            _seasonal_kernel(θ.l_seasonal, θ.period, θ.bandwidth, kernels.seasonal)
        k_noise = θ.σ_noise^2 * WhiteKernel()
        return @gppp let
            x̄ = GP(k_trend)
            x̃ = GP(k_seasonal)
            ϵ = GP(k_noise)
            y = x̄ + x̃ + ϵ
        end
    elseif has_seasonal
        k_seasonal =
            θ.σ_seasonal^2 *
            _seasonal_kernel(θ.l_seasonal, θ.period, θ.bandwidth, kernels.seasonal)
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
