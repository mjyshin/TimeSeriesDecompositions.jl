const _FilterModel = Union{X11, STL}

# Decomposition of a filter model from y, its trend and seasonal on the grid t of spacing Δt
function _filter_decomposition(model::_FilterModel, r::_Report, y, t, Δt, period, trend,
        seasonal, smoother)
    noise = y - seasonal - trend
    rate = _trend_rate(trend, Δt)
    grid = (; t, trend, seasonal, noise, rate)
    θ, fit_info = _finish(r, nameof(typeof(model)), Δt, grid)
    return Decomposition(model, t, Float64(period), Component(trend), Component(seasonal),
        Component(noise), Component(rate), θ, smoother, fit_info)
end

"""
    predict(d::Decomposition{<:Union{X11,STL}}, t_new=d.t) -> Decomposition

Evaluate a filter decomposition at times `t_new` within the observed range by interpolating
the fitted components: the trend by its not-a-knot cubic spline (the rate by the spline's
derivative), the seasonal trigonometrically through the cycle of samples around each time,
and the noise as `NaN` off the observed times.
"""
function predict(d::Decomposition{<:_FilterModel}, t_new::AbstractVector{<:Real} = d.t)
    (; Δt, grid) = d._fit_info
    trend, seasonal, noise, rate = _interpolate_components(
        grid, Δt, round(Int, d.period / Δt), t_new)
    return Decomposition(d.model, collect(Float64, t_new), d.period, Component(trend),
        Component(seasonal), Component(noise), Component(rate), d.θ, d.smoother,
        d._fit_info)
end

function _interpolate_components(grid::NamedTuple, Δt::Float64, p::Int, t_new)
    (; t, trend, seasonal, noise, rate) = grid
    _check_in_range(t, t_new; atol = 1e-8Δt)
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

function Base.show(io::IO, ::MIME"text/plain", d::Decomposition{<:_FilterModel})
    _show_header(io, d)
    _show_line(io, "components", "trend, seasonal, noise, rate (point estimates)")
    _show_line(io, "period", _fmt(d.period))
    stages = [stage => f.model for (stage, f) in pairs(d.smoother)]
    (; estimated, derived, criteria) = d._fit_info
    row(k) = k => (k ∈ derived ? "$(_fmt(d.θ[k])) (derived)" : d.θ[k])
    sections = (
        "smoothers" => stages,
        "estimated" => [row(k) for k in keys(d.θ) if k ∈ estimated],
        "fixed" => [row(k) for k in keys(d.θ) if k ∉ estimated],
        "criteria" => collect(pairs(criteria))
    )
    width = _section_width(last.(sections)...)
    for (label, rows) in sections
        _show_section(io, label, rows; width)
    end
    return nothing
end
