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
Component(mean::AbstractVector{<:Real}) =
    Component(Float64.(mean), fill(NaN, length(mean)), Vector{Float64}[], Vector{Float64}[], Float64[])

# Restrict a component to the time indices idx
_subset(c::Component, idx) =
    Component(c.mean[idx], c.std[idx], [l[idx] for l in c.lower], [u[idx] for u in c.upper], c.sig)
_subset(::Nothing, idx) = nothing


#== Helpers for filter-based (uniformly sampled) decompositions ==#

# Sort (y, t) by t and check uniform spacing; returns (y, t, Δt)
function _uniform_grid(y::AbstractVector{<:Real}, t::AbstractVector{<:Real})
    length(y) == length(t) || throw(DimensionMismatch("y and t must have the same length"))
    length(t) ≥ 2 || throw(ArgumentError("need at least 2 observations"))
    perm = sortperm(t)
    y_s, t_s = Float64.(y[perm]), Float64.(t[perm])
    Δ = diff(t_s)
    Δt = first(Δ)
    Δt > 0 && all(δ -> isapprox(δ, Δt; rtol=1e-8), Δ) ||
        throw(ArgumentError("filter-based decompositions require uniformly spaced, distinct t"))
    any(isnan, y_s) && throw(ArgumentError("y must not contain NaN"))
    return y_s, t_s, Δt
end

# Period in units of t → integer number of samples per cycle
function _samples_per_period(period::Real, Δt::Real)
    isfinite(period) && period > 0 || throw(DomainError(period, "period must be positive and finite"))
    T = period / Δt
    isapprox(T, round(T); rtol=1e-8) && round(Int, T) ≥ 2 ||
        throw(ArgumentError("period / Δt = $T must be an integer ≥ 2"))
    return round(Int, T)
end

# Odd window length (in samples) → half-width h
function _halfwidth(window::Integer, name::AbstractString)
    window ≥ 3 && isodd(window) || throw(ArgumentError("$name must be an odd integer ≥ 3, got $window"))
    return (window - 1) ÷ 2
end

_nextodd(x::Real) = (n = ceil(Int, x); isodd(n) ? n : n + 1)

# Indices of t_new in the uniform grid t (error if any t_new is off-grid)
function _grid_indices(t::AbstractVector{Float64}, Δt::Float64, t_new::AbstractVector{<:Real})
    map(t_new) do τ
        k = (τ - first(t)) / Δt
        i = round(Int, k) + 1
        isapprox(k, i - 1; atol=1e-8) && 1 ≤ i ≤ length(t) || throw(ArgumentError(
            "t = $τ is not an in-sample time; filter-based decompositions cannot extrapolate or interpolate"))
        i
    end
end

# Rate dx̄/dt from a uniformly sampled trend, differentiated over its finite (contiguous)
# span only so that NaN ends (e.g. a symmetric filter) do not propagate
function _trend_rate(x̄::AbstractVector{<:Real}, Δt::Real)
    r = fill(NaN, length(x̄))
    idx = findall(isfinite, x̄)
    length(idx) ≥ 3 && idx == first(idx):last(idx) || return r
    r[idx] = Filters.derivative(x̄[idx]) ./ Δt
    return r
end
