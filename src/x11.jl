"""
    X11Decomp <: AbstractDecomp

X-11-style seasonal-trend decomposition from classical filters: a Henderson trend
filter with asymmetric (minimum-revision or direct) end weights and S(n,m) seasonal
filters. See `fit(X11Decomp, ...)`. Point estimates only (`std` is `NaN`).
"""
struct X11Decomp <: AbstractDecomp
    t::Vector{Float64}
    trend::Component
    seasonal::Component
    noise::Component
    rate::Component
    θ::NamedTuple
end

const _x11_end_filters = (mrf=Filters.mrf, daf=Filters.daf, henderson=Filters.henderson)

function _x11_trend_filter(name::Symbol, ic_ratio)
    haskey(_x11_end_filters, name) || throw(ArgumentError(
        "unknown trend filter :$name; valid: $(keys(_x11_end_filters))"))
    name === :mrf && ic_ratio !== nothing || return _x11_end_filters[name]
    # Musgrave parameterisation (Proietti & Luati 2008, §4.1): U = [1], Z = [j],
    # δ₁²/σ² = 4/(πR²) with R the I/C ratio
    ic_ratio > 0 || throw(DomainError(ic_ratio, "ic_ratio must be positive"))
    δ = 2 / (ic_ratio * sqrt(π))
    return (y, h; kwargs...) -> Filters.mrf(y, h; kwargs..., dU=1, δ=δ)
end

"""
    fit(X11Decomp, y, t=eachindex(y); period, kwargs...) -> X11Decomp

X-11-style decomposition of `y` observed at uniformly spaced times `t` (wraps
`Filters.x11`). Two passes: an initial trend with the `init_end_filter` and S(n′,m′)
seasonal filter, then the final trend with `end_filter` and S(n,m) seasonal filter.

# Keyword arguments
- `period`: seasonal period in units of `t` (required); `period / Δt` must be an integer.
- `trend_window`: odd Henderson filter length in samples (default 13, i.e. `h = 6`).
- `end_filter`: asymmetric end weights of the final trend filter: `:mrf` (minimum
  revision, default), `:daf` (direct asymmetric) or `:henderson` (none; ends are `NaN`).
- `ic_ratio`: I/C ratio `R` for the Musgrave form of the `:mrf` end weights
  (`δ₁²/σ² = 4/(πR²)`; X-11 uses 3.5 for the 13-term filter). `nothing` (default) keeps
  the `Filters.mrf` defaults.
- `init_trend_window`, `init_end_filter`: as above for the first pass
  (defaults `trend_window`, `:daf`).
- `seasonal_filters`: `((n′, m′), (n, m))` for the S(n,m) seasonal filters of the two
  passes (default `((3, 3), (3, 5))`, i.e. S3×3 then S3×5).

The rate is the derivative of the trend (cubic-spline, one-sided at the ends) per unit
of `t`. Component estimates carry no uncertainty (`std` is `NaN`).
"""
function fit(
    ::Type{X11Decomp},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    period::Real,
    trend_window::Integer=13,
    end_filter::Symbol=:mrf,
    ic_ratio::Union{Nothing,Real}=nothing,
    init_trend_window::Integer=trend_window,
    init_end_filter::Symbol=:daf,
    seasonal_filters::NTuple{2,NTuple{2,Integer}}=((3, 3), (3, 5)),
)
    y_s, t_s, Δt = _uniform_grid(y, t)
    T  = _samples_per_period(period, Δt)
    h  = _halfwidth(trend_window, "trend_window")
    h′ = _halfwidth(init_trend_window, "init_trend_window")
    N  = length(y_s)
    N ≥ max(2h + 1, 2h′ + 1, 2T) || throw(ArgumentError(
        "need at least max(trend_window, init_trend_window, 2·period/Δt) = $(max(2h+1, 2h′+1, 2T)) observations, got $N"))

    (n′, m′), (n, m) = seasonal_filters
    comp = Filters.x11(y_s, h, T;
        Ft=_x11_trend_filter(end_filter, ic_ratio), n, m,
        h′, Ft′=_x11_trend_filter(init_end_filter, nothing), n′, m′,
    )
    rate = _trend_rate(comp.x̄, Δt)

    θ = (
        period=Float64(period), Δt=Δt, trend_window=Int(trend_window), end_filter=end_filter,
        ic_ratio=ic_ratio, init_trend_window=Int(init_trend_window),
        init_end_filter=init_end_filter, seasonal_filters=seasonal_filters,
    )
    return X11Decomp(t_s, Component(comp.x̄), Component(comp.x̃), Component(comp.ϵ), Component(rate), θ)
end
