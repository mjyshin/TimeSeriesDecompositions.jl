"""
    AbstractDecomp

Supertype of the seasonal-trend decompositions [`GPDecomp`](@ref), [`X11Decomp`](@ref) and
[`STLDecomp`](@ref), with fields `t`, `trend`, `seasonal`, `noise`, `rate` (each a
[`Component`](@ref) or `nothing`) and the hyperparameters `θ`. Fit with `fit(D, y, t; ...)`,
evaluate at other times with `predict(d, t_new)`; `d[:name]` is `d.name`.
"""
abstract type AbstractDecomp end

Base.getindex(d::AbstractDecomp, name::Symbol) = getproperty(d, name)

"""
    Component

Estimate of one component at the times `d.t`: `mean`, `std`, and credible-interval bounds
`lower[i]`, `upper[i]` at level `sig[i]`. Point estimates (`X11Decomp`, `STLDecomp`) have
`std` all `NaN` and no intervals. `c[:name]` is `c.name`.
"""
struct Component
    mean::Vector{Float64}
    std::Vector{Float64}
    lower::Vector{Vector{Float64}}
    upper::Vector{Vector{Float64}}
    sig::Vector{Float64}
end

Base.getindex(c::Component, name::Symbol) = getproperty(c, name)

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

function _show_header(io::IO, d::AbstractDecomp)
    t1, tN = extrema(d.t)
    return print(io, "$(nameof(typeof(d))){N=$(length(d.t)), t∈[$t1, $tN]}")
end

Base.show(io::IO, d::AbstractDecomp) = _show_header(io, d)

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
