## Plot recipe for decompositions ##

# Components in panel order; the index is also the series color, whatever is shown
const _COMPONENTS = (:trend, :seasonal, :noise, :rate)
const _TITLES = (trend = "Trend", seasonal = "Seasonal", noise = "Noise", rate = "Rate")
# LaTeX guides as plain "\$…\$" strings, which Plots renders without LaTeXStrings
const _GUIDES = (trend = "\$\\bar x(t)\$", seasonal = "\$\\tilde x(t)\$",
    noise = "\$\\epsilon(t)\$", rate = "\$r(t)\$")

# Not `levels`, which Plots reserves for contours and validates on every series
const _INTERVALS = [0.5, 0.9]

# The credible levels of the bands, widest first: `true` the default, `false`, `()` or `0` none
_levels(intervals::Bool) = intervals ? _levels(_INTERVALS) : Float64[]
function _levels(intervals)
    lv = sort(unique(Float64[x for x in intervals]); rev = true)
    all(0 .≤ lv .< 1) || throw(DomainError(intervals, "intervals must lie in [0, 1)"))
    return filter(>(0), lv)    # a level of 0 is the mean alone, drawn anyway
end

# `plot(d)` draws every component and `plot(d, :trend)` one
@recipe f(d::Decomposition) = (d, _COMPONENTS)
@recipe f(d::Decomposition, component::Symbol) = (d, (component,))

@recipe function f(d::Decomposition, components::Union{Tuple, AbstractVector{Symbol}};
        intervals = _INTERVALS)
    for v in components
        v ∈ _COMPONENTS ||
            throw(ArgumentError("unknown component :$v; valid: $(_COMPONENTS)"))
    end
    levels = _levels(intervals)
    shown = [v for v in unique(components) if !isnothing(d[v])]
    isempty(shown) &&
        throw(ArgumentError("the decomposition has none of $(Tuple(components))"))
    z = [quantile(Normal(), (1 + p) / 2) for p in levels]
    n = length(shown)

    layout := (n, 1)
    legend --> false
    size --> (600, 200n)
    for (i, v) in enumerate(shown)
        c = d[v]
        color = findfirst(==(v), _COMPONENTS)
        if any(isfinite, c.std)
            for ζ in z
                @series begin
                    subplot := i
                    seriescolor := color
                    linealpha := 0
                    fillalpha --> 1 / (length(levels) + 1)
                    ribbon := ζ .* c.std
                    label := ""
                    d.t, c.mean
                end
            end
        end
        @series begin
            subplot := i
            seriescolor := color
            linewidth --> 2
            title --> _TITLES[v]
            yguide --> _GUIDES[v]
            i == n && (xguide --> "\$t\$")
            d.t, c.mean
        end
    end
end
