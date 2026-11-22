## Plot recipe for decompositions ##

# Components in panel order; the index is also the series color, whatever is shown
const _COMPONENTS = (:trend, :seasonal, :noise, :rate)
const _TITLES = (trend = "Trend", seasonal = "Seasonal", noise = "Noise", rate = "Rate")
# LaTeX guides as plain "\$…\$" strings, which Plots renders without LaTeXStrings
const _GUIDES = (trend = "\$\\bar x(t)\$", seasonal = "\$\\tilde x(t)\$",
    noise = "\$\\epsilon(t)\$", rate = "\$r(t)\$")

@recipe function f(d::Decomposition; vars = _COMPONENTS, sig = [0.5, 0.9])
    vars = vars isa Symbol ? (vars,) : vars
    for v in vars
        v ∈ _COMPONENTS ||
            throw(ArgumentError("unknown component :$v; valid: $(_COMPONENTS)"))
    end
    levels = sort(unique(Float64.(vec([sig;]))); rev = true)    # widest band first
    all(0 .< levels .< 1) || throw(DomainError(sig, "sig levels must be in (0, 1)"))
    shown = [v for v in unique(vars) if !isnothing(d[v])]
    isempty(shown) && throw(ArgumentError("the decomposition has none of $(Tuple(vars))"))
    z = [quantile(Normal(), (1 + p) / 2) for p in levels]
    n = length(shown)

    layout := (n, 1)
    legend --> false
    size --> (800, 160 + 140n)
    for (i, v) in enumerate(shown)
        c = d[v]
        color = findfirst(==(v), _COMPONENTS)
        if any(isfinite, c.std)
            for ζ in z
                @series begin
                    subplot := i
                    seriescolor := color
                    linealpha := 0
                    fillalpha --> 1 / 3
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
