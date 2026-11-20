"""
    LocalPolynomial(; bandwidth=nothing, degree=3, kernel=HendersonKernel(),
                    endpoints=MinimumRevision()) <: AbstractTrendSmoother

Local polynomial trend filter [1] for regularly spaced `t` of spacing `Δ`: at each time, the
`kernel`-weighted least-squares polynomial of degree `degree` through the observations
within `bandwidth / 2`, evaluated there. At the observed times it is the linear filter of
X-11; between them, the same fit at the continuous offsets.

- `bandwidth`: window width `2hΔ` in units of `t` (`h ≥ 1` an integer), or a vector of
  candidates, of which the one minimizing the mean leave-one-out cross-validation score
  over the interior is chosen. `nothing` gives `12Δ`, or inside a decomposition is derived
  from the period.
- `degree`: degree of the local polynomial.
- `kernel`: an [`AbstractSmootherKernel`](@ref).
- `endpoints`: an [`AbstractEndpoints`](@ref) for the first and last `h` samples.

`predict(f, t_new)` evaluates the smoother at times in `[t₁, t_N]`. `f.state.weights` holds
the weights at offsets `-h:h`, and `f.state.end_weights[q + 1]` those at offsets `-h:q`.

[1] T. Proietti and A. Luati. "Real time estimation in local polynomial regression, with
application to trend-cycle analysis". Annals of Applied Statistics 2 (4), 2008.
"""
struct LocalPolynomial{K <: AbstractSmootherKernel, E <: AbstractEndpoints} <:
       AbstractTrendSmoother
    bandwidth::Union{Nothing, Float64, Vector{Float64}}
    degree::Int
    kernel::K
    endpoints::E
end

function LocalPolynomial(;
        bandwidth::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        degree::Integer = 3,
        kernel::AbstractSmootherKernel = HendersonKernel(),
        endpoints::AbstractEndpoints = MinimumRevision()
)
    degree ≥ 0 || throw(DomainError(degree, "degree must be non-negative"))
    isnothing(bandwidth) ||
        all(>(0), bandwidth) ||
        throw(DomainError(bandwidth, "bandwidth must be positive"))
    bandwidth = bandwidth isa AbstractVector ? collect(Float64, bandwidth) :
                _float(bandwidth)
    return LocalPolynomial(bandwidth, Int(degree), kernel, endpoints)
end

"""
    Henderson(; bandwidth=nothing, endpoints=MinimumRevision())

The Henderson filter: a local cubic [`LocalPolynomial`](@ref) with the
[`HendersonKernel`](@ref). As a type, `Henderson` is `LocalPolynomial{HendersonKernel}`.
"""
const Henderson = LocalPolynomial{HendersonKernel}

function Henderson(;
        bandwidth::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        endpoints::AbstractEndpoints = MinimumRevision()
)
    return LocalPolynomial(; bandwidth, degree = 3, kernel = HendersonKernel(), endpoints)
end

_name(m::LocalPolynomial) = _is_henderson(m) ? :Henderson : :LocalPolynomial
_is_henderson(m::LocalPolynomial) = m.kernel isa HendersonKernel && m.degree == 3

_fmt_bandwidth(b) = isnothing(b) ? "nothing" : b isa Real ? _fmt(b) : string(_fmt.(b))

function Base.show(io::IO, m::LocalPolynomial)
    bandwidth = _fmt_bandwidth(m.bandwidth)
    if _is_henderson(m)
        return print(io, "Henderson(bandwidth=$bandwidth, endpoints=$(m.endpoints))")
    end
    return print(
        io,
        "LocalPolynomial(bandwidth=$bandwidth, degree=$(m.degree), ",
        "kernel=$(m.kernel), endpoints=$(m.endpoints))"
    )
end

function _fit(model::LocalPolynomial, y::AbstractVector{<:Real}, t::AbstractVector{<:Real})
    y, t = _sorted_data(y, t)
    Δ = _require_grid(t, "LocalPolynomial")
    estimated = (;)
    bandwidth = something(model.bandwidth, 2_DEFAULT_HALFWIDTH * Δ)
    if bandwidth isa AbstractVector
        bandwidth, cv_score = _select_bandwidth(y, bandwidth, Δ, model.degree, model.kernel)
        estimated = (; bandwidth, cv_score)
    end
    h = _halfwidth(bandwidth, Δ)
    length(y) ≥ 2h + 1 || throw(
        ArgumentError(
        "series of length $(length(y)) is shorter than the $(2h + 1) samples of " *
        "bandwidth $bandwidth",
    ),
    )
    2h > model.degree ||
        throw(ArgumentError("bandwidth $bandwidth is too short for degree $(model.degree)"))
    g = _grid_kernel(model.kernel, h)
    endpoints, criteria = _resolve_endpoints(model.endpoints, g, y, model.degree)
    estimated = (; estimated..., criteria...)
    resolved = LocalPolynomial(bandwidth, model.degree, model.kernel, endpoints)
    return _fitted_grid(resolved, g, endpoints, model.degree, y, t, Δ, estimated)
end

function predict(f::FittedSmoother{<:LocalPolynomial}, t_new::AbstractVector{<:Real})
    return _predict_grid(f, t_new)
end

const _DEFAULT_HALFWIDTH = 6

# Half-width h in samples of a bandwidth 2hΔ
function _halfwidth(bandwidth::Real, Δ::Real)
    h = bandwidth / 2Δ
    isapprox(h, round(h); rtol = 1e-8) && round(h) ≥ 1 || throw(
        ArgumentError(
        "bandwidth / Δt = $(bandwidth / Δ) must be an even integer ≥ 2, got " *
        "bandwidth = $bandwidth",
    ),
    )
    return round(Int, h)
end

# Bandwidth minimizing the mean leave-one-out CV score (P&L §5) over the interior points
function _select_bandwidth(y, bandwidths, Δ, degree, kernel)
    isempty(bandwidths) && throw(ArgumentError("no candidate bandwidths"))
    N = length(y)
    halfwidths = [_halfwidth(b, Δ) for b in bandwidths]
    scores = map(halfwidths) do h
        N ≥ 2h + 1 && 2h > degree || return Inf
        w = _local_weights(_grid_kernel(kernel, h), (-h):h, degree)
        return mean(((h + 1):(N - h))) do t
            return ((y[t] - dot(w, view(y, (t - h):(t + h)))) / (1 - w[h + 1]))^2
        end
    end
    all(isinf, scores) && throw(ArgumentError("no candidate bandwidth fits the series"))
    i = argmin(scores)
    return bandwidths[i], scores[i]
end

## Grid weights, shared by LocalPolynomial and MovingAverage ##

# Weight κ(d) at offset d = (tᵢ - x) / Δ, positive up to the integer offset `reach`
struct _GridKernel{F}
    κ::F
    reach::Int
end

function _grid_kernel(kernel::AbstractSmootherKernel, h::Integer)
    return _GridKernel(d -> _kernel_value(kernel, d, h), Int(h))
end

_scale(g::_GridKernel) = max(g.reach, 1)

function _design(d::AbstractVector, degree::Integer, s::Real)
    return [(dᵢ / s)^k for dᵢ in d, k in 0:degree]
end

_unit(n::Integer) = [1.0; zeros(n - 1)]

# w = KX(XᵀKX)⁻¹e₁ (P&L eqs. 3–4), with offsets scaled by the reach for conditioning
function _local_weights(g::_GridKernel, d::AbstractVector, degree::Integer)
    X = _design(d, degree, _scale(g))
    KX = g.κ.(d) .* X
    return KX * ((X' * KX) \ _unit(degree + 1))
end

_end_weights(::NoEndpoints, _, dp, _, _) = fill(NaN, length(dp))
_end_weights(::DirectAsymmetric, g, dp, _, degree) = _local_weights(g, dp, degree)

# Minimum-revision weights on the available offsets dp, missing df: the KKT system of
# P&L eqs. 8–9 with D = σ²I and bias column Z = dᵈ*⁺¹
function _end_weights(e::MinimumRevision{Float64}, g, dp, df, degree)
    isinf(e.ratio) &&    # P&L eq. 12
        return _end_weights(MinimumRevision(e.degree + 1, 0.0), g, dp, df, degree)
    d = [dp; df]
    p, f = eachindex(dp), (length(dp) + 1):length(d)
    s = _scale(g)
    w = _local_weights(g, d, degree)
    U = _design(d, e.degree, s)
    z = @. (d / s)^(e.degree + 1)
    ρ = e.ratio * Float64(s)^(2 * (e.degree + 1))    # for the scaled column z
    Up, zp = U[p, :], z[p]
    zf_wf = dot(view(z, f), view(w, f))
    k = e.degree + 1
    A = [(I + ρ * zp * zp') Up; Up' zeros(k, k)]
    b = [ρ * zf_wf * zp; U[f, :]' * view(w, f)]
    return w[p] + (A \ b)[p]
end

function _validate(::DirectAsymmetric, reach, degree)
    reach ≥ degree || throw(
        ArgumentError(
        "direct asymmetric end weights of degree $degree need h ≥ $degree, got $reach",
    ),
    )
    return nothing
end

function _validate(e::MinimumRevision, reach, degree)
    e.degree < max(degree, 1) || throw(
        ArgumentError(
        "minimum-revision degree ($(e.degree)) must be below the smoother degree " *
        "($degree)",
    ),
    )
    e.degree < reach || throw(
        ArgumentError(
        "minimum-revision degree $(e.degree) needs h > $(e.degree), got h = $reach"
    ),
    )
    e.ratio == Inf &&
        !_allows_infinite_ratio(e, reach) &&
        throw(
            ArgumentError(
            "ratio=Inf reproduces degree $(e.degree + 1), which needs h > $(e.degree + 1)",
        ),
        )
    return nothing
end

_validate(::NoEndpoints, _, _) = nothing

_allows_infinite_ratio(e::MinimumRevision, reach::Integer) = e.degree + 1 < reach

# Validated endpoints, with the ratio estimated if it is nothing, and the criteria
function _resolve_endpoints(e::AbstractEndpoints, g::_GridKernel, y, degree)
    _validate(e, g.reach, degree)
    e isa MinimumRevision{Nothing} || return e, (;)
    return _estimate_ratio(e, g, y, degree)
end

const _MIN_MSRE_POINTS = 10

# Ratio minimizing the mean square revision error of the real-time weights (P&L §5): grid
# search over {0, Inf} ∪ 10^(-4:0.1:2), refined by Brent's method on log₁₀ρ
function _estimate_ratio(e::MinimumRevision{Nothing}, g::_GridKernel, y, degree)
    r = g.reach
    N = length(y)
    ts = (r + 1):(N - r)
    if length(ts) < _MIN_MSRE_POINTS
        note = "too few points to estimate the ratio by MSRE; using 0"
        return MinimumRevision(e.degree, 0.0), (; ratio = 0.0, note_ratio = note)
    end
    w = _local_weights(g, (-r):r, degree)
    final = [dot(w, view(y, (t - r):(t + r))) for t in ts]
    function msre(ρ)
        v = _end_weights(MinimumRevision(e.degree, ρ), g, (-r):0, 1:r, degree)
        return mean(abs2(final[i] - dot(v, view(y, (t - r):t))) for (i, t) in enumerate(ts))
    end

    log_grid = -4:0.1:2
    grid = [0.0; exp10.(log_grid); _allows_infinite_ratio(e, r) ? [Inf] : Float64[]]
    scores = msre.(grid)
    i = argmin(scores)
    ratio, score = grid[i], scores[i]
    if 1 < i ≤ length(log_grid) + 1
        lo, hi = log_grid[max(i - 2, 1)], log_grid[min(i, length(log_grid))]
        res = optimize(s -> msre(exp10(s)), lo, hi, Brent())
        if Optim.minimum(res) < score
            ratio, score = exp10(Optim.minimizer(res)), Optim.minimum(res)
        end
    end
    return MinimumRevision(e.degree, ratio), (; ratio, msre = score)
end

function _fitted_grid(model, g, endpoints, degree, y, t, Δ, estimated)
    r = g.reach
    N = length(y)
    N ≥ 2r + 1 ||
        throw(ArgumentError("series of length $N is too short for the bandwidth"))
    weights = _local_weights(g, (-r):r, degree)
    end_weights = Vector{Float64}[_end_weights(endpoints, g, (-r):q, (q + 1):r, degree)
                                  for q in 0:(r - 1)]
    values = fill(NaN, N)
    for i in (r + 1):(N - r)
        values[i] = dot(weights, view(y, (i - r):(i + r)))
    end
    for q in 0:(r - 1)
        v = end_weights[q + 1]
        values[N - q] = dot(v, view(y, (N - q - r):N))
        values[1 + q] = dot(reverse(v), view(y, 1:(1 + q + r)))
    end
    state = (; kernel = g, endpoints, degree, Δ, weights, end_weights)
    return FittedSmoother(model, t, y, values, state, estimated)
end

function _predict_grid(f::FittedSmoother, t_new::AbstractVector{<:Real})
    _check_in_range(f.t, t_new)
    (; kernel, endpoints, degree, Δ) = f.state
    N = length(f.y)
    values = Vector{Float64}(undef, length(t_new))
    for (j, x) in enumerate(t_new)
        c = (x - first(f.t)) / Δ + 1    # fractional index of x
        i = round(Int, c)
        values[j] = if isapprox(c, i; atol = 1e-9)
            f.values[i]
        else
            _evaluate(kernel, endpoints, degree, f.y, f.values, c, N)
        end
    end
    return values
end

# Local fit at the fractional index c. Near the ends, minimum-revision estimates are
# interpolated between grid points: their off-grid extension jumps where observations enter.
function _evaluate(g::_GridKernel, endpoints, degree, y, values, c, N)::Float64
    r = g.reach
    inside(k) = g.κ(k - c) > 0    # grid index k has positive weight at x
    idx = filter(inside, max(1, floor(Int, c) - r):min(N, ceil(Int, c) + r))
    d = idx .- c
    after = [k - c for k in (N + 1):(ceil(Int, c) + r) if inside(k)]
    before = [k - c for k in (floor(Int, c) - r):0 if inside(k)]
    y_idx = view(y, idx)
    isempty(after) && isempty(before) && return dot(_local_weights(g, d, degree), y_idx)
    isempty(after) ||
        isempty(before) ||
        throw(ArgumentError("the series is too short for the bandwidth"))
    endpoints isa MinimumRevision && !isinteger(c) && return _lagrange(values, c)
    isempty(after) && return dot(_end_weights(endpoints, g, -d, -before, degree), y_idx)
    return dot(_end_weights(endpoints, g, d, after, degree), y_idx)
end

function _lagrange(v, c)
    k = clamp(floor(Int, c), 2, length(v) - 2)
    nodes = (k - 1):(k + 2)
    return sum(nodes) do i
        return v[i] * prod((c - j) / (i - j) for j in nodes if j != i)
    end
end
