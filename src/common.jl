"""
    AbstractDecomp

Supertype of all seasonal-trend decompositions (`GPDecomp`, `X11Decomp`, `STLDecomp`).

Every subtype has the fields `t`, `trend`, `seasonal`, `noise`, `rate` (each a
[`Component`](@ref), `seasonal`/`noise` possibly `nothing`) and `θ`, a `NamedTuple` of
the method's (fitted or resolved) parameters. Construct with `fit(D, y, t; ...)` and
re-evaluate with `predict(d, t_new)`.
"""
abstract type AbstractDecomp end

"""
    Component

Estimate of one component at the times `d.t`: `mean`, `std`, and credible-interval
bounds `lower[i]`/`upper[i]` at level `sig[i]`.

Methods without uncertainty quantification (`X11Decomp`, `STLDecomp`) return point
estimates: `std` is all `NaN` and `lower`, `upper` and `sig` are empty.
"""
struct Component
    mean::Vector{Float64}
    std::Vector{Float64}
    lower::Vector{Vector{Float64}}
    upper::Vector{Vector{Float64}}
    sig::Vector{Float64}
end

# Point estimate without uncertainty
function Component(mean::AbstractVector{<:Real})
    return Component(
        Float64.(mean),
        fill(NaN, length(mean)),
        Vector{Float64}[],
        Vector{Float64}[],
        Float64[],
    )
end

# Restrict a component to the time indices idx
function _subset(c::Component, idx)
    return Component(
        c.mean[idx], c.std[idx], [l[idx] for l in c.lower], [u[idx] for u in c.upper], c.sig
    )
end
_subset(::Nothing, idx) = nothing
