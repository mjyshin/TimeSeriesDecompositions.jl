const _DEFAULT_KERNEL = Matern52Kernel()

# Resolve the `kernel` keyword to (trend, seasonal=(effect, envelope)); a kernel at any level
# applies to every component below it, and unspecified components take the default
_resolve_kernels(k::Kernel) = (trend = k, seasonal = (effect = k, envelope = k))
_resolve_kernels(k::NamedTuple) = _resolve_kernels(_DEFAULT_KERNEL, k)
function _resolve_kernels(default::Kernel, k::NamedTuple)
    _check_keys(k, (:trend, :seasonal), "kernel")
    trend = get(k, :trend, default)
    trend isa Kernel || throw(ArgumentError("kernel.trend must be a Kernel"))
    return (
        trend = trend, seasonal = _resolve_seasonal(get(k, :seasonal, default), default))
end
function _resolve_kernels(k)
    throw(ArgumentError("kernel must be a Kernel or a NamedTuple, got $(typeof(k))"))
end

_resolve_seasonal(k::Kernel, _) = (effect = k, envelope = k)
function _resolve_seasonal(k::NamedTuple, default::Kernel)
    _check_keys(k, (:effect, :envelope), "kernel.seasonal")
    effect, envelope = get(k, :effect, default), get(k, :envelope, default)
    effect isa Kernel && envelope isa Kernel ||
        throw(ArgumentError("kernel.seasonal.effect and kernel.seasonal.envelope must be Kernels"))
    return (effect = effect, envelope = envelope)
end
function _resolve_seasonal(k, _)
    throw(ArgumentError("kernel.seasonal must be a Kernel or a NamedTuple, got $(typeof(k))"))
end

const _default_kernels = _resolve_kernels(_DEFAULT_KERNEL)

# Periodic kernel without its constant Fourier coefficient a₀, rescaled to unit variance:
# (k - a₀) / (1 - a₀). Dropping a non-negative coefficient keeps it positive semi-definite.
struct _ZeroMeanKernel{K <: Kernel, T <: Real} <: Kernel
    kernel::K
    a₀::T
end

(k::_ZeroMeanKernel)(x, y) = (k.kernel(x, y) - k.a₀) / (1 - k.a₀)

_zero_mean(k::_ZeroMeanKernel, K) = (K .- k.a₀) ./ (1 - k.a₀)

function KernelFunctions.kernelmatrix(k::_ZeroMeanKernel, x::AbstractVector)
    return _zero_mean(k, kernelmatrix(k.kernel, x))
end
function KernelFunctions.kernelmatrix(k::_ZeroMeanKernel, x::AbstractVector, y::AbstractVector)
    return _zero_mean(k, kernelmatrix(k.kernel, x, y))
end
function KernelFunctions.kernelmatrix_diag(k::_ZeroMeanKernel, x::AbstractVector)
    return _zero_mean(k, kernelmatrix_diag(k.kernel, x))
end
function KernelFunctions.kernelmatrix_diag(
        k::_ZeroMeanKernel, x::AbstractVector, y::AbstractVector)
    return _zero_mean(k, kernelmatrix_diag(k.kernel, x, y))
end

# a₀ by the trapezoid rule over one period; aliasing overestimates it by O(n⁻²) for Matérn-1/2
function _zero_mean_periodic(k_per::Kernel, period::Real; n::Int = 1024)
    u = collect((0:(n - 1)) .* (period / n))
    return _ZeroMeanKernel(k_per, mean(kernelmatrix(k_per, [0.0], u)))
end

# Envelope of lengthscale bandwidth = period / ρ; ρ = 0 is the static model
function _seasonal_kernel(l_seasonal::Real, period::Real, ρ::Real, kernels::NamedTuple)
    k_per = with_lengthscale(kernels.effect, l_seasonal) ∘ PeriodicTransform(1 / period)
    k_per = _zero_mean_periodic(k_per, period)
    iszero(ρ) && return k_per
    return k_per * (kernels.envelope ∘ ScaleTransform(ρ / period))
end

# Kernels of the components (`nothing` if absent) in standardized units
function _component_kernels(θ::NamedTuple, kernels::NamedTuple = _default_kernels)
    trend = θ.σ_trend^2 * with_lengthscale(kernels.trend, θ.l_trend)
    seasonal = if isfinite(θ.period)
        θ.σ_seasonal^2 * _seasonal_kernel(θ.l_seasonal, θ.period, θ.ρ, kernels.seasonal)
    end
    noise = θ.σ_noise > 0 ? θ.σ_noise^2 * WhiteKernel() : nothing
    return (; trend, seasonal, noise)
end

# Kernel of the observations y = trend + seasonal + noise
function _observation_kernel(ks::NamedTuple)
    k = ks.trend
    isnothing(ks.seasonal) || (k += ks.seasonal)
    isnothing(ks.noise) || (k += ks.noise)
    return k
end
