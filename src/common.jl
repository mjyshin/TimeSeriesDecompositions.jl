"""
    AbstractDecompositionModel

Supertype of the seasonal-trend decomposition models [`GaussianProcess`](@ref),
[`X11`](@ref) and [`STL`](@ref): immutable structs of settings, built with keyword
constructors. `fit(model, y, t; period)` returns a [`Decomposition`](@ref).
"""
abstract type AbstractDecompositionModel end

"""
    Component

Estimate of one component at the times `d.t`: its posterior `mean` and `std`, a normal
marginal at each time, whose quantiles [`quantile`](@ref quantile(::Component, ::Real)) gives.
Point estimates ([`X11`](@ref), [`STL`](@ref)) have `std` all `NaN`. `c[:name]` is `c.name`.
"""
struct Component
    mean::Vector{Float64}
    std::Vector{Float64}
end

Base.getindex(c::Component, name::Symbol) = getproperty(c, name)

"""
    quantile(c::Component, p)

The quantile `p` of the component's normal marginal at each time, `mean + std * Φ⁻¹(p)`;
several levels `p` give a times × levels matrix. The central interval at coverage `level` is
`quantile(c, [(1 - level) / 2, (1 + level) / 2])`. Point estimates give `NaN`.
"""
function Statistics.quantile(c::Component, p::Real)
    0 ≤ p ≤ 1 || throw(DomainError(p, "p must lie in [0, 1]"))
    return c.mean .+ c.std .* quantile(Normal(), p)
end
function Statistics.quantile(c::Component, p)
    reduce(hcat, [quantile(c, q) for q in p];
        init = Matrix{Float64}(undef, length(c.mean), 0))
end

"""
    Decomposition{M<:AbstractDecompositionModel}

A decomposition fitted by `fit(model::M, y, t; period)`, with fields `model`, `t`, `period`,
`trend`, `seasonal`, `noise`, `rate` (each a [`Component`](@ref) or `nothing`), the
hyperparameters `θ` and, for filter models, the fitted stage smoothers `smoother`.
The `rate` is the trend's rate of change dx̄/dt, in units of `y` per unit of `t`; for
log-transformed data, it is the exponential growth rate.
Evaluate at other times with `predict(d, t_new)`; `d[:name]` is `d.name`.

With Plots.jl loaded, `plot(d)` draws one panel per component, `plot(d, :trend)` one, and
`plot(d, [:trend, :rate])` (or a tuple) those given, in that order, skipping those `d` does
not have: the mean as a line and, where the `std` is finite, a band `mean ± zσ` per credible
level in the keyword `intervals` (default `[0.5, 0.9]`; `true` the default, `false`, `()` or
`0` the mean alone; `z` the normal quantile at `(1 + level) / 2`), each with opacity
1/(number of levels + 1) so that they stack. The colors are fixed whatever is shown (trend 1,
seasonal 2, noise 3, rate 4); other Plots attributes pass through.
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
Component(mean::AbstractVector{<:Real}) = Component(Float64.(mean), fill(NaN, length(mean)))

# Restrict a component to the time indices idx
function _subset(c::Component, idx)
    return Component(c.mean[idx], c.std[idx])
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
