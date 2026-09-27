const _FilterDecomp = Union{X11Decomp,STLDecomp}

"""
    predict(d::Union{X11Decomp,STLDecomp}, t_new=d.t) -> typeof(d)

Restrict a filter-based decomposition to the in-sample times `t_new`. Filters cannot
extrapolate or interpolate, so every element of `t_new` must be one of `d.t`.
"""
function predict(d::_FilterDecomp, t_new::AbstractVector{<:Real}=d.t)
    idx = _grid_indices(d.t, d.θ.Δt, t_new)
    return typeof(d)(
        d.t[idx],
        _subset(d.trend, idx),
        _subset(d.seasonal, idx),
        _subset(d.noise, idx),
        _subset(d.rate, idx),
        d.θ,
    )
end

function Base.show(io::IO, d::_FilterDecomp)
    N = length(d.t)
    t1, tN = extrema(d.t)
    println(io, "$(nameof(typeof(d))){N=$N, t∈[$t1, $tN]}")
    println(io, "  components : trend, seasonal, noise, rate (point estimates)")
    println(io, "  period     : $(d.θ.period)")
    print(io, "  filters    :")
    for (name, f) in pairs(d.θ.filters)
        print(io, "\n    ", rpad(string(name), 21), repr(f))
    end
    isempty(d.θ.estimated) && return nothing
    print(io, "\n  estimated  :")
    for (name, value) in pairs(d.θ.estimated)
        print(io, "\n    ", rpad(string(name), 21), value)
    end
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", d::_FilterDecomp) = show(io, d)
