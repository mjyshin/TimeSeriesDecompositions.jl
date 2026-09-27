const _FilterDecomp = Union{X11Decomp,STLDecomp}

"""
    predict(d::Union{X11Decomp,STLDecomp}, t_new=d.t) -> typeof(d)

Evaluate a filter-based decomposition at times `t_new` within the observed range. At the
observed times the fitted components are returned as they are; in between they are
interpolated, which adds no information:

- the trend is the not-a-knot cubic spline through the fitted trend and the rate its
  derivative, as at the observed times (both `NaN` where the trend is undefined). This
  differs slightly from re-evaluating the trend smoother,
  `predict(d.θ.smoothers.trend, t_new)`, whose derivative jumps at the samples;
- the seasonal is the trigonometric interpolation across phases of the fitted seasonal:
  the trigonometric polynomial of degree ≤ p/2 through the cycle of samples around each
  time (for odd `p`, the average of the two nearly centred cycles). A fixed seasonal
  pattern is interpolated exactly. In the interior of the series, for the `S(n × m)`
  filters of [`X11Decomp`](@ref), this equals those filters applied at an intermediate
  phase to the trigonometric interpolation of the detrended series; near the ends,
  where the filters are truncated and the cycles shifted inward, it is only
  approximately so. It assumes a smooth within-cycle
  pattern: a sharp seasonal peak overshoots between the samples;
- the noise is `NaN`, since nothing is observed there, so `trend + seasonal + noise`
  recovers `y` only at the observed times.

Times beyond the observed range throw a `DomainError`: filter-based decompositions do not
forecast.
"""
function predict(d::_FilterDecomp, t_new::AbstractVector{<:Real}=d.t)
    trend, seasonal, noise, rate = _interpolate_components(
        d.θ.grid, d.θ.Δt, round(Int, d.θ.period / d.θ.Δt), t_new
    )
    return typeof(d)(
        collect(Float64, t_new),
        Component(trend),
        Component(seasonal),
        Component(noise),
        Component(rate),
        d.θ,
    )
end

# The components at the times t_new, from their values at the grid times
function _interpolate_components(grid::NamedTuple, Δt::Float64, p::Int, t_new)
    (; t, trend, seasonal, noise, rate) = grid
    _check_in_range(t, t_new; atol=1e-8Δt)
    defined = any(isfinite, trend) ? _finite_range(trend) : (1:0)
    spline = length(defined) ≥ 3 ? _spline(trend[defined]) : nothing
    n = length(t_new)
    out = (zeros(n), zeros(n), fill(NaN, n), zeros(n))
    for (j, τ) in enumerate(t_new)
        c = _grid_position(first(t), Δt, τ)
        if isinteger(c)
            i = Int(c)
            for (v, x) in zip(out, (trend, seasonal, noise, rate))
                v[j] = x[i]
            end
            continue
        end
        out[2][j] = _periodic_interpolation(seasonal, c, p)
        if isnothing(spline) || !(first(defined) < c < last(defined))
            out[1][j] = out[4][j] = NaN
        else
            u = c - first(defined) + 1
            out[1][j] = spline(u)
            out[4][j] = Interpolations.gradient(spline, u)[1] / Δt
        end
    end
    return out
end

function Base.show(io::IO, ::MIME"text/plain", d::_FilterDecomp)
    _show_header(io, d)
    _show_line(io, "components", "trend, seasonal, noise, rate (point estimates)")
    _show_line(io, "period", _fmt(d.θ.period))
    _show_section(io, "smoothers", pairs(map(f -> f.model, d.θ.smoothers)))
    _show_section(io, "estimated", pairs(d.θ.estimated))
    return nothing
end
