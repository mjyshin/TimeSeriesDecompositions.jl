const _DEFAULT_KERNEL = Matern52Kernel()

# Resolve the `kernel` keyword to (trend, seasonal=(effect, envelope)); a kernel at any level
# applies to every component below it, and unspecified components take the default
_resolve_kernels(k::Kernel) = _resolve_kernels(k, (;))
_resolve_kernels(k::NamedTuple) = _resolve_kernels(_DEFAULT_KERNEL, k)
function _resolve_kernels(default::Kernel, k::NamedTuple)
    _check_keys(k, (:trend, :seasonal), "kernel")
    trend = get(k, :trend, default)
    trend isa Kernel || throw(ArgumentError("kernel.trend must be a Kernel"))
    kernels = (
        trend = trend, seasonal = _resolve_seasonal(get(k, :seasonal, default), default))
    return (trend = _unit_variance(kernels.trend),
        seasonal = map(_unit_variance, kernels.seasonal))
end

# The likelihood evaluates kernels as functions of a Euclidean distance (the seasonal one on
# the periodic embedding in the plane), and the σ hyperparameters carry the scale
function _unit_variance(k::Kernel)
    stationary = try
        all(isapprox(k(0.0, d), k(x, x + d)) && isapprox(k(0.0, d), k(x + d, x)) &&
                isapprox(k(0.0, d), k([x, 1.0], [x + 0.6d, 1.0 + 0.8d]))
        for d in (0.7, 1.9), x in (-2.3, 4.1))
    catch
        false
    end
    stationary && k(0.0, 0.0) > 0 || throw(ArgumentError(
        "kernels must be stationary and isotropic functions of scalar times, got $k"))
    c = k(0.0, 0.0)
    return c == 1 ? k : inv(c) * k
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

# Kernels of the components (`nothing` if absent) and the noise variance, independent
# across observations, in standardized units
function _component_kernels(θ::NamedTuple, kernels::NamedTuple = _default_kernels)
    trend = θ.σ_trend^2 * with_lengthscale(kernels.trend, θ.l_trend)
    seasonal = if isfinite(θ.period)
        θ.σ_seasonal^2 * _seasonal_kernel(θ.l_seasonal, θ.period, θ.ρ, kernels.seasonal)
    end
    return (; trend, seasonal, noise = θ.σ_noise^2)
end

# Kernel of the signal trend + seasonal; the observations add the noise variance
function _signal_kernel(ks::NamedTuple)
    return isnothing(ks.seasonal) ? ks.trend : ks.trend + ks.seasonal
end

## Likelihood on the training times: a stationary covariance takes one value per distinct
## lag, so it and its derivatives are computed lag-wise

# A stationary kernel at distance r, with zero (not NaN) derivatives at r = 0
_at(k::Kernel, r) = iszero(r) ? k(0.0, 0.0) * one(r) : k(0.0, r)

struct _LagIndex
    lags::Vector{Float64}
    index::Matrix{Int}
end

function _LagIndex(t::AbstractVector{Float64})
    D = abs.(t .- t')
    lags = unique(D)
    return _LagIndex(lags, Int.(indexin(D, lags)))
end

# Signal covariance at each lag in standardized units, as `_signal_kernel` gives, followed
# by the noise variance
function _lag_covariance(θ::NamedTuple, lags, kernels::NamedTuple; n::Int = 1024)
    P = θ.period
    has_seasonal = isfinite(P)
    ke, kv = kernels.seasonal.effect, kernels.seasonal.envelope
    effect(d) = _at(ke, 2abs(sinpi(d / P)) / θ.l_seasonal)
    a₀ = has_seasonal ? mean(effect, (0:(n - 1)) .* (P / n)) : zero(θ.σ_trend)
    signal = map(lags) do d
        v = θ.σ_trend^2 * _at(kernels.trend, d / θ.l_trend)
        if has_seasonal
            s = (effect(d) - a₀) / (1 - a₀)
            θ.ρ == 0 || (s *= _at(kv, d * θ.ρ / P))
            v += θ.σ_seasonal^2 * s
        end
        return v
    end
    return [signal; θ.σ_noise^2]
end

# Negative log marginal likelihood from lag-wise covariances `v`, and with `gradient`, the
# lag-wise sums of (K⁻¹ - ααᵀ)/2, its derivative with respect to `v`
function _nll_lags(v::AbstractVector{Float64}, lag::_LagIndex, y, gradient::Bool)
    K = v[lag.index]
    K[diagind(K)] .+= v[end] + _JITTER
    C = cholesky!(Symmetric(K); check = false)
    issuccess(C) || throw(PosDefException(C.info))
    α = C \ y
    nll = (dot(y, α) + logdet(C) + length(y) * log(2π)) / 2
    gradient || return nll, nothing
    W = inv(C)
    w = zeros(length(v))
    for (k, i) in pairs(lag.index)
        w[i] += (W[k] - α[k[1]] * α[k[2]]) / 2
    end
    w[end] = sum(i -> W[i, i] - α[i]^2, eachindex(α)) / 2
    return nll, w
end

function _nll(θ::NamedTuple, lag::_LagIndex, y, kernels::NamedTuple)
    return first(_nll_lags(_lag_covariance(θ, lag.lags, kernels), lag, y, false))
end
