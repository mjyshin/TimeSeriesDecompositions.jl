"""
    AbstractFilterKernel

Kernel `κⱼ` weighting the observations `yₜ₊ⱼ`, `j = -h:h`, in the local polynomial fit of
a [`LocalPolynomialFilter`](@ref). Subtypes: [`HendersonKernel`](@ref),
[`BiweightKernel`](@ref), [`TricubeKernel`](@ref), [`EpanechnikovKernel`](@ref) and
[`UniformKernel`](@ref). Evaluate with [`kernel_weights`](@ref).
"""
abstract type AbstractFilterKernel end

"""
    HendersonKernel()

Henderson (1916) kernel `κⱼ = [(h+1)² - j²][(h+2)² - j²][(h+3)² - j²]`. With a local
cubic it gives the weights minimising the variance of the third differences of the
filtered series: the Henderson filter of X-11.
"""
struct HendersonKernel <: AbstractFilterKernel end

"""
    BiweightKernel()

Biweight kernel `κⱼ = (1 - u²)²`, `u = j / (h + 1)`.
"""
struct BiweightKernel <: AbstractFilterKernel end

"""
    TricubeKernel()

Tricube kernel `κⱼ = (1 - |u|³)³`, `u = j / (h + 1)`, as in loess.
"""
struct TricubeKernel <: AbstractFilterKernel end

"""
    EpanechnikovKernel()

Epanechnikov kernel `κⱼ = 1 - u²`, `u = j / (h + 1)`.
"""
struct EpanechnikovKernel <: AbstractFilterKernel end

"""
    UniformKernel()

Uniform kernel `κⱼ = 1`.
"""
struct UniformKernel <: AbstractFilterKernel end

"""
    kernel_weights(kernel::AbstractFilterKernel, h::Integer) -> Vector{Float64}

Unnormalised weights `κⱼ`, `j = -h:h`, of `kernel` for a filter of length `2h + 1`.
Kernels other than [`HendersonKernel`](@ref) are evaluated at `u = j / (h + 1)`, so that
the outermost weights are non-zero.
"""
function kernel_weights(kernel::AbstractFilterKernel, h::Integer)
    h ≥ 0 || throw(DomainError(h, "h must be non-negative"))
    return [_profile(kernel, j / (h + 1)) for j in (-h):h]
end

function kernel_weights(::HendersonKernel, h::Integer)
    h ≥ 0 || throw(DomainError(h, "h must be non-negative"))
    return Float64[
        ((h + 1)^2 - j^2) * ((h + 2)^2 - j^2) * ((h + 3)^2 - j^2) for j in (-h):h
    ]
end

_profile(::BiweightKernel, u) = (1 - u^2)^2
_profile(::TricubeKernel, u) = (1 - abs(u)^3)^3
_profile(::EpanechnikovKernel, u) = 1 - u^2
_profile(::UniformKernel, u) = one(u)
