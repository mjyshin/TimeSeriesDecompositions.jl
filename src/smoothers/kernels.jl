"""
    AbstractSmootherKernel

Weight `κ(d)` of the observation at offset `d` samples in a [`LocalPolynomial`](@ref) of
half-width `h`, zero for `|d| ≥ h + 1`. Evaluate with [`kernel_weights`](@ref).
"""
abstract type AbstractSmootherKernel end

"""
    HendersonKernel()

Henderson kernel `κⱼ = [(h+1)² - j²][(h+2)² - j²][(h+3)² - j²]` [1]; with a local cubic, the
Henderson filter of X-11.

[1] R. Henderson. "Note on graduation by adjusted average". Transactions of the Actuarial
Society of America 17, 1916.
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

Uniform kernel `κⱼ = 1`; off the grid, `κ(d) = clamp(h + 1 - |d|, 0, 1)`, the overlap of
the unit cell at `d` with the window.
"""
struct UniformKernel <: AbstractSmootherKernel end

"""
    kernel_weights(kernel::AbstractSmootherKernel, h::Integer) -> Vector{Float64}

Return the unnormalized weights `κⱼ`, `j = -h:h`, of a filter of length `2h + 1`.
"""
function kernel_weights(kernel::AbstractSmootherKernel, h::Integer)
    h ≥ 0 || throw(DomainError(h, "h must be non-negative"))
    return [_kernel_value(kernel, j, h) for j in (-h):h]
end

function _kernel_value(::HendersonKernel, d::Real, h::Integer)
    abs(d) < h + 1 || return 0.0
    return Float64(((h + 1)^2 - d^2) * ((h + 2)^2 - d^2) * ((h + 3)^2 - d^2))
end

_kernel_value(::UniformKernel, d::Real, h::Integer) = clamp(h + 1 - abs(d), 0.0, 1.0)

function _kernel_value(kernel::AbstractSmootherKernel, d::Real, h::Integer)
    abs(d) < h + 1 || return 0.0
    return Float64(_profile(kernel, d / (h + 1)))
end

_profile(::BiweightKernel, u) = (1 - u^2)^2
_profile(::TricubeKernel, u) = (1 - abs(u)^3)^3
_profile(::EpanechnikovKernel, u) = 1 - u^2
_profile(::UniformKernel, u) = one(u)
