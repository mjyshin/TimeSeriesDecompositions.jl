"""
    STLDecomp <: AbstractDecomp

Seasonal-trend decomposition with a loess trend. See `fit(STLDecomp, ...)`.
Point estimates only (`std` is `NaN`).

!!! warning "Placeholder"
    This is **not yet** the STL procedure of Cleveland et al. (1990): there is no
    cycle-subseries smoothing, low-pass filter, inner/outer loop or robustness
    weighting. It currently removes a moving-average trend, estimates a *fixed*
    seasonal pattern (stable seasonal filter) and smooths the deseasonalised series
    with loess (`Filters.decompose` with `Ft=Filters.loess`).
"""
struct STLDecomp <: AbstractDecomp
    t::Vector{Float64}
    trend::Component
    seasonal::Component
    noise::Component
    rate::Component
    θ::NamedTuple
end

"""
    fit(STLDecomp, y, t=eachindex(y); period, trend_window) -> STLDecomp

Loess-based seasonal-trend decomposition of `y` observed at uniformly spaced times `t`.
**Placeholder** for STL; see [`STLDecomp`](@ref).

# Keyword arguments
- `period`: seasonal period in units of `t` (required); `period / Δt` must be an integer.
- `trend_window`: odd loess window length in samples. The trend is a local-quadratic
  (degree 2; STL uses degree 1) Loess.jl fit with span `trend_window / N`.
  Defaults to Cleveland et al.'s rule `n_t ≥ 1.5 n_p / (1 − 1.5 / n_s)` with the
  smallest seasonal window `n_s = 7`.

The rate is the derivative of the trend (cubic-spline, one-sided at the ends) per unit
of `t`. Component estimates carry no uncertainty (`std` is `NaN`).
"""
function fit(
    ::Type{STLDecomp},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Real,
    trend_window::Union{Nothing,Integer}=nothing,
)
    y_s, t_s, Δt = _uniform_grid(y, t)
    T = _samples_per_period(period, Δt)
    trend_window = something(trend_window, _nextodd(1.5T / (1 - 1.5 / 7)))
    h = _halfwidth(trend_window, "trend_window")
    N = length(y_s)
    N ≥ max(2h + 1, 2T) || throw(ArgumentError(
        "need at least max(trend_window, 2·period/Δt) = $(max(2h+1, 2T)) observations, got $N"))

    comp = Filters.decompose(y_s, h; Ft=Filters.loess, seasonal=true, T)
    rate = _trend_rate(comp.x̄, Δt)

    θ = (period=Float64(period), Δt=Δt, trend_window=Int(trend_window))
    return STLDecomp(t_s, Component(comp.x̄), Component(comp.x̃), Component(comp.ϵ), Component(rate), θ)
end
