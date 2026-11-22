"""
    AbstractDecompositionModel

Supertype of the seasonal-trend decomposition models [`GaussianProcess`](@ref),
[`X11`](@ref) and [`STL`](@ref): immutable structs of settings, built with keyword
constructors. `fit(model, y, t; period)` returns a [`Decomposition`](@ref).
"""
abstract type AbstractDecompositionModel end

"""
    Component

Estimate of one component at the times `d.t`: `mean`, `std`, and credible-interval bounds
`lower[i]`, `upper[i]` at level `sig[i]`. Point estimates ([`X11`](@ref), [`STL`](@ref))
have `std` all `NaN` and no intervals. `c[:name]` is `c.name`.
"""
struct Component
    mean::Vector{Float64}
    std::Vector{Float64}
    lower::Vector{Vector{Float64}}
    upper::Vector{Vector{Float64}}
    sig::Vector{Float64}
end

Base.getindex(c::Component, name::Symbol) = getproperty(c, name)

"""
    Decomposition{M<:AbstractDecompositionModel}

A decomposition fitted by `fit(model::M, y, t; period)`, with fields `model`, `t`, `period`,
`trend`, `seasonal`, `noise`, `rate` (each a [`Component`](@ref) or `nothing`), the
hyperparameters `θ` and, for filter models, the fitted stage smoothers `smoother`.
Evaluate at other times with `predict(d, t_new)`; `d[:name]` is `d.name`.
"""
struct Decomposition{M <: AbstractDecompositionModel}
    model::M
    t::Vector{Float64}
    period::Float64
    trend::Component
    seasonal::Union{Component, Nothing}
    noise::Union{Component, Nothing}
    rate::Component
    θ::NamedTuple
    smoother::Union{NamedTuple, Nothing}
    _fit_info::NamedTuple
end

Base.getindex(d::Decomposition, name::Symbol) = getproperty(d, name)

"""
    fit(M::Type{<:AbstractDecompositionModel}, y, t=eachindex(y); period, kwargs...)

Shorthand for `fit(M(; kwargs...), y, t; period)`.
"""
function fit(::Type{M}, y::AbstractVector{<:Real},
        t::AbstractVector{<:Real} = eachindex(y);
        period::Union{Real, Nothing} = nothing, kwargs...) where {
        M <: AbstractDecompositionModel}
    model = M(; kwargs...)
    return isnothing(period) ? fit(model, y, t) : fit(model, y, t; period)
end

# Show a model by the fields that differ from its defaults
function Base.show(io::IO, m::M) where {M <: AbstractDecompositionModel}
    default = Base.typename(M).wrapper()
    changed = ("$k=$(getfield(m, k))"
    for k in fieldnames(M) if !isequal(getfield(m, k), getfield(default, k)))
    return print(io, nameof(M), "(", join(changed, ", "), ")")
end

# Point estimate without uncertainty
function Component(mean::AbstractVector{<:Real})
    return Component(
        Float64.(mean),
        fill(NaN, length(mean)),
        Vector{Float64}[],
        Vector{Float64}[],
        Float64[]
    )
end

# Restrict a component to the time indices idx
function _subset(c::Component, idx)
    return Component(
        c.mean[idx], c.std[idx], [l[idx] for l in c.lower], [u[idx] for u in c.upper], c.sig
    )
end
_subset(::Nothing, idx) = nothing

function _check_keys(nt::NamedTuple, valid, name)
    for key in keys(nt)
        key ∈ valid || throw(ArgumentError("Unknown $name key :$key. Valid keys: $valid"))
    end
    return nothing
end

## Display ##

_fmt(x::AbstractFloat) = isfinite(x) ? string(round(x; sigdigits = 4)) : string(x)
_fmt(x) = string(x)

function _show_header(io::IO, d::Decomposition)
    t1, tN = isempty(d.t) ? (NaN, NaN) : extrema(d.t)
    name = nameof(typeof(d.model))
    return print(io, "Decomposition{$name}(N=$(length(d.t)), t∈[$t1, $tN])")
end

Base.show(io::IO, d::Decomposition) = _show_header(io, d)

_show_line(io::IO, label, value) = print(io, "\n  ", rpad(label, 11), ": ", value)

function _section_width(sections...)
    widths = (textwidth(string(name)) + 1 for rows in sections for (name, _) in rows)
    return max(21, maximum(widths; init = 0))
end

function _show_section(io::IO, label, rows; width = _section_width(rows))
    isempty(rows) && return nothing
    print(io, "\n  ", rpad(label, 11), ":")
    for (name, value) in rows
        print(io, "\n    ", rpad(string(name), width), _fmt(value))
    end
    return nothing
end
