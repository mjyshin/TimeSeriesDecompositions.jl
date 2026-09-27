"""
    AbstractSmootherKernel

Kernel `κ(d)` weighting the observation at offset `d` (in samples, not necessarily an
integer) in the local polynomial fit of a [`LocalPolynomial`](@ref) of half-width `h`;
it vanishes for `|d| ≥ h + 1`. Subtypes: [`HendersonKernel`](@ref),
[`BiweightKernel`](@ref), [`TricubeKernel`](@ref), [`EpanechnikovKernel`](@ref) and
[`UniformKernel`](@ref). Evaluate with [`kernel_weights`](@ref).
"""
abstract type AbstractSmootherKernel end

"""
    HendersonKernel()

Henderson (1916) kernel `κⱼ = [(h+1)² - j²][(h+2)² - j²][(h+3)² - j²]`. With a local
cubic it gives the weights minimising the variance of the third differences of the
filtered series: the Henderson filter of X-11.
"""
struct HendersonKernel <: AbstractSmootherKernel end

"""
    BiweightKernel()

Biweight kernel `κⱼ = (1 - u²)²`, `u = j / (h + 1)`.
"""
struct BiweightKernel <: AbstractSmootherKernel end

"""
    TricubeKernel()

Tricube kernel `κⱼ = (1 - |u|³)³`, `u = j / (h + 1)`, as in loess.
"""
struct TricubeKernel <: AbstractSmootherKernel end

"""
    EpanechnikovKernel()

Epanechnikov kernel `κⱼ = 1 - u²`, `u = j / (h + 1)`.
"""
struct EpanechnikovKernel <: AbstractSmootherKernel end

"""
    UniformKernel()

Uniform kernel `κⱼ = 1` at the offsets of the window; between grid points, an observation
at offset `d` has the overlap `clamp(h + 1 - |d|, 0, 1)` of its unit cell with the
window, so that it enters continuously.
"""
struct UniformKernel <: AbstractSmootherKernel end

"""
    kernel_weights(kernel::AbstractSmootherKernel, h::Integer) -> Vector{Float64}

Unnormalised weights `κⱼ`, `j = -h:h`, of `kernel` for a filter of length `2h + 1`: the
kernel's continuous form `κ(d)` (see [`AbstractSmootherKernel`](@ref)) at the integer
offsets.
"""
function kernel_weights(kernel::AbstractSmootherKernel, h::Integer)
    h ≥ 0 || throw(DomainError(h, "h must be non-negative"))
    return [_kernel_value(kernel, j, h) for j in (-h):h]
end

# Continuous kernel at offset d (in samples) for half-width h: zero for |d| ≥ h + 1.
# The Henderson kernel is Henderson's polynomial in d; the others are evaluated at
# u = d / (h + 1).
function _kernel_value(::HendersonKernel, d::Real, h::Integer)
    abs(d) < h + 1 || return 0.0
    return Float64(((h + 1)^2 - d^2) * ((h + 2)^2 - d^2) * ((h + 3)^2 - d^2))
end

# The uniform kernel is the overlap of the unit cell around d with the window, so that
# observations enter continuously between grid points; at integer offsets it is 1
_kernel_value(::UniformKernel, d::Real, h::Integer) = clamp(h + 1 - abs(d), 0.0, 1.0)

function _kernel_value(kernel::AbstractSmootherKernel, d::Real, h::Integer)
    abs(d) < h + 1 || return 0.0
    return Float64(_profile(kernel, d / (h + 1)))
end

_profile(::BiweightKernel, u) = (1 - u^2)^2
_profile(::TricubeKernel, u) = (1 - abs(u)^3)^3
_profile(::EpanechnikovKernel, u) = 1 - u^2
_profile(::UniformKernel, u) = one(u)
