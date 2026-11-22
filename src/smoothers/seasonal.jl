"""
    StableSeasonal() <: AbstractSeasonalSmoother

Fixed seasonal pattern: the mean of the observations at each phase of the cycle.
"""
struct StableSeasonal <: AbstractSeasonalSmoother end

"""
    SeasonalMovingAverage(; n=3, m=5) <: AbstractSeasonalSmoother

The `S(n × m)` seasonal filter of X-11: an `m`-term then an `n`-term moving average (`n`,
`m` odd) of each cycle-subseries. At the ends of a subseries the averages are truncated and
renormalized, unlike X-11's published asymmetric weights.
"""
struct SeasonalMovingAverage <: AbstractSeasonalSmoother
    n::Int
    m::Int
end

function SeasonalMovingAverage(; n::Integer = 3, m::Integer = 5)
    n ≥ 1 && m ≥ 1 && isodd(n) && isodd(m) ||
        throw(ArgumentError("S(n × m) needs odd, positive n and m, got ($n, $m)"))
    return SeasonalMovingAverage(Int(n), Int(m))
end

"""
    SeasonalLOESS(; bandwidth=nothing, degree=1) <: AbstractSeasonalSmoother

The seasonal smoother of STL [1]: a [`LOESS`](@ref) of each cycle-subseries, fitted at each
cycle to the `bandwidth / period + 1` nearest cycles (Cleveland et al.'s `n_s`), so that the
window is `bandwidth` wide away from the ends.

- `bandwidth`: window width in units of `t`, rounded up to an even number of periods (at
  least `degree + 2`). `nothing` in the final seasonal stage of [`STL`](@ref) or
  [`X11`](@ref) chooses it from the data by X-11's moving seasonality ratio: the window
  whose interior weights reduce the irregular's variance as much as the `S3×3`, `S3×5` or
  `S3×9` filter that the ratio selects (6, 8 and 14 periods at degree 1). Elsewhere, and
  below seven cycles, `nothing` gives six periods, the seven cycles that Cleveland et al.
  recommend as a minimum.
- `degree`: `0`, `1` or `2`.

Inside [`STL`](@ref), each subseries fit is also evaluated one cycle beyond each end. A
subseries shorter than the window widens it as [`LOESS`](@ref) does for a span above 1, by
`√(n_s / m)` for `m` cycles; STL's Fortran adds `(n_s - m) / 2` cycles instead, so the two
differ only there.

[1] R. B. Cleveland, W. S. Cleveland, J. E. McRae and I. Terpenning. "STL: a seasonal-trend
decomposition procedure based on loess". Journal of Official Statistics 6 (1), 1990.
"""
struct SeasonalLOESS <: AbstractSeasonalSmoother
    bandwidth::Union{Nothing, Float64}
    degree::Int
end

function SeasonalLOESS(; bandwidth::Union{Nothing, Real} = nothing, degree::Integer = 1)
    isnothing(bandwidth) ||
        bandwidth > 0 ||
        throw(DomainError(bandwidth, "bandwidth must be positive"))
    degree ∈ (0, 1, 2) || throw(DomainError(degree, "degree must be 0, 1 or 2"))
    return SeasonalLOESS(_float(bandwidth), Int(degree))
end

Base.show(io::IO, ::StableSeasonal) = print(io, "StableSeasonal()")
function Base.show(io::IO, m::SeasonalLOESS)
    return print(io, "SeasonalLOESS(bandwidth=$(_fmt_bandwidth(m.bandwidth)), ",
        "degree=$(m.degree))")
end
function Base.show(io::IO, m::SeasonalMovingAverage)
    return print(io, "SeasonalMovingAverage(n=$(m.n), m=$(m.m))")
end

# Cycles needed for one symmetric estimate
_min_cycles(m::SeasonalMovingAverage) = m.n + m.m - 1
_min_cycles(::StableSeasonal) = 1
_min_cycles(::SeasonalLOESS) = 1

function _fit(
        model::AbstractSeasonalSmoother,
        y::AbstractVector{<:Real},
        t::AbstractVector{<:Real};
        period::Real
)
    y, t = _sorted_data(y, t)
    Δ = _require_grid(t, nameof(typeof(model)))
    p = _samples_per_period(period, Δ)
    model = _resolve_seasonal(model, period)
    values = _seasonal_values(model, y, p, period)
    return FittedSmoother(model, t, y, values, (; period = Float64(period), Δ), (;))
end

"""
    predict(f::FittedSmoother{<:AbstractSeasonalSmoother}, t_new) -> Vector{Float64}

Evaluate the fitted seasonal pattern at times `t_new` within the observed range,
interpolating trigonometrically between phases, as in `predict` of an [`X11`](@ref) fit.
"""
function predict(
        f::FittedSmoother{<:AbstractSeasonalSmoother}, t_new::AbstractVector{<:Real}
)
    (; period, Δ) = f.state
    _check_in_range(f.t, t_new; atol = 1e-8Δ)
    p = round(Int, period / Δ)
    return map(t_new) do τ
        c = _grid_position(first(f.t), Δ, τ)
        return isinteger(c) ? f.values[Int(c)] : _periodic_interpolation(f.values, c, p)
    end
end

# Trigonometric interpolation at the fractional index c through p consecutive values around
# c (the two nearly centered windows averaged for odd p), clamped into the series
function _periodic_interpolation(v::AbstractVector{Float64}, c::Real, p::Int)
    k = floor(Int, c)
    if iseven(p)
        return _window_interpolation(v, c, p, k - p ÷ 2 + 1)
    end
    return (
        _window_interpolation(v, c, p, k - (p - 1) ÷ 2) +
        _window_interpolation(v, c, p, k - (p - 3) ÷ 2)
    ) / 2
end

function _window_interpolation(v, c, p, first_index)
    lo = clamp(first_index, 1, length(v) - p + 1)
    return sum(v[j] * _periodic_cardinal(c - j, p) for j in lo:(lo + p - 1))
end

# Periodic cardinal (Dirichlet) function: 1 at multiples of p, 0 at other integers
function _periodic_cardinal(x::Real, p::Int)
    s = sinpi(x / p)
    abs(s) < 1e-12 && return 1.0
    return isodd(p) ? sinpi(x) / (p * s) : sinpi(x) / (p * s) * cospi(x / p)
end

function _seasonal_values(::StableSeasonal, x, p, _)
    s = similar(x)
    for phase in 1:p
        idx = phase:p:length(x)
        s[idx] .= mean(view(x, idx))
    end
    return s
end

function _seasonal_values(model::SeasonalMovingAverage, x, p, _)
    inner, outer = fill(1 / model.m, model.m), fill(1 / model.n, model.n)
    s = similar(x)
    for phase in 1:p
        idx = phase:p:length(x)
        s[idx] = _truncated_filter(outer, _truncated_filter(inner, view(x, idx)))
    end
    return s
end

function _seasonal_values(model::SeasonalLOESS, x, p, period)
    q = _cycles(model, period)
    s = similar(x)
    for phase in 1:p
        idx = phase:p:length(x)
        s[idx] = _subseries_loess(x[idx], q, model.degree, 1:length(idx))
    end
    return s
end

# Seasonal smoother with hyperparameters left as nothing resolved from the period
_resolve_seasonal(model::AbstractSeasonalSmoother, _) = model

function _resolve_seasonal(model::SeasonalLOESS, period)
    periods = 6
    if !isnothing(model.bandwidth)
        n = max(1, ceil(Int, model.bandwidth / period - 1e-8))
        periods = iseven(n) ? n : n + 1
    end
    # interior fits weight periods - 1 cycles, at least degree + 1
    periods ≥ model.degree + 2 || throw(
        ArgumentError("SeasonalLOESS of degree $(model.degree) needs a bandwidth of at " *
                      "least $(model.degree + 2) periods"),
    )
    return SeasonalLOESS(periods * period, model.degree)
end

# Hyperparameters of a seasonal smoother derived from the period
_seasonal_derived(model::SeasonalLOESS) = isnothing(model.bandwidth) ? (:bandwidth,) : ()
_seasonal_derived(_) = ()

# Cycles in the window of a resolved SeasonalLOESS
_cycles(model::SeasonalLOESS, period) = round(Int, model.bandwidth / period) + 1

# Sum of squared weights of a filter of each subseries away from its ends
function _interior_sum_squares(m::SeasonalMovingAverage)
    w = zeros(m.n + m.m - 1)
    for i in 1:(m.n), j in 1:(m.m)

        w[i + j - 1] += 1 / (m.n * m.m)
    end
    return sum(abs2, w)
end

function _interior_sum_squares(m::SeasonalLOESS, period)
    q = _cycles(m, period)
    cycles = collect(1.0:(2q + 1))
    wts = ones(2q + 1)
    idx, λ, c = _local_fit(cycles, wts, q / (2q + 1), 0.0, m.degree, q + 1.0)
    return sum(abs2, _local_weight(cycles, wts, q + 1.0, λ, c, i) for i in idx)
end

# SeasonalLOESS of the even number of periods (at least degree + 2, at most 40) whose
# interior weights have the sum of squares closest to that of the S(n × m) filter f: the
# same reduction of the irregular's variance
function _matched_loess(f::SeasonalMovingAverage, degree, period)
    target = _interior_sum_squares(f)
    candidates = (SeasonalLOESS(n * period, degree) for n in (2cld(degree + 2, 2)):2:40)
    return argmin(m -> abs(_interior_sum_squares(m, period) - target), candidates)
end

# Loess with q neighbors of the subseries x at cycles 1, …, m, evaluated at the cycles ks
function _subseries_loess(x, q, degree, ks)
    m = length(x)
    cycles, wts = collect(1.0:m), ones(m)
    return [_local_value(cycles, x, wts, q / m, 0.0, degree, Float64(k)) for k in ks]
end

# Seasonal smoothing of x and one cycle beyond each end, at indices 1 - p, …, N + p: the
# subseries loess evaluated there, or the nearest value of the same phase
function _extended_seasonal(model::SeasonalLOESS, x, p, period)
    q = _cycles(model, period)
    N = length(x)
    c = zeros(N + 2p)
    for phase in 1:p
        m = length(phase:p:N)
        c[phase:p:(phase + (m + 1) * p)] = _subseries_loess(
            x[phase:p:N], q, model.degree, 0:(m + 1))
    end
    return c
end

function _extended_seasonal(model::AbstractSeasonalSmoother, x, p, period)
    s = _seasonal_values(model, x, p, period)
    N = length(x)
    return [s[i < 1 ? i + p : i > N ? i - p : i] for i in (1 - p):(N + p)]
end

# Centered filter, truncated and renormalized at the ends
function _truncated_filter(w::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    h = (length(w) - 1) ÷ 2
    N = length(y)
    return map(1:N) do t
        js = max(-h, 1 - t):min(h, N - t)
        wt = view(w, js .+ (h + 1))
        return dot(wt, view(y, t .+ js)) / sum(wt)
    end
end
