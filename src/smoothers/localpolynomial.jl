"""
    LocalPolynomial(; bandwidth=nothing, degree=3, kernel=HendersonKernel(),
                    endpoints=MinimumRevision()) <: AbstractTrendSmoother

(`bandwidth=nothing` means `12Δ` (13 samples) when fitted on its own, and is derived from
the period inside a decomposition.)

Local polynomial trend smoother for regularly spaced times `t` of spacing `Δ`: at each
time `x`, the weighted least-squares fit of a polynomial of degree `degree` to the
observations within `bandwidth / 2` of `x`, weighted by `kernel` at their offsets
`(tᵢ - x) / Δ`, evaluated at `x`. The bandwidth is twice the radius: with
`bandwidth = 2hΔ`, the `2h + 1` samples at offsets `-h:h` are in the window. At the
observed times this is the linear filter of X-11 and Proietti & Luati (2008); with the
continuous form of the kernel it is also defined between them, where it reduces to the
filter at every grid point.

- `bandwidth`: in units of `t`, an even multiple `2hΔ` of the spacing with `h ≥ 1`, or a
  vector of candidates from which the one minimizing the mean leave-one-out
  cross-validation score `mean(((yₜ - m̂ₜ) / (1 - w₀))²)` over the interior points is
  chosen. `nothing` gives `12Δ`, the conventional 13-term filter for monthly data;
  inside a decomposition it is instead derived from the period.
- `degree`: degree of the local polynomial.
- `kernel`: an [`AbstractSmootherKernel`](@ref).
- `endpoints`: an [`AbstractEndpoints`](@ref) for the first and last `h` samples.

[`Henderson`](@ref) is the local cubic with the Henderson kernel.

`fit(model, y, t)` requires regularly spaced `t`, choosing any hyperparameter left as
`nothing` (reported with an `@info` message and in `f.estimated`); `predict(f, t_new)`
evaluates the smoother at any times in `[t₁, t_N]`. `f.state.weights` and
`f.state.end_weights[q + 1]` hold the weights for offsets `-h:h` and, at the point with
`q` later observations, `-h:q`.
"""
struct LocalPolynomial{K<:AbstractSmootherKernel,E<:AbstractEndpoints} <:
       AbstractTrendSmoother
    bandwidth::Union{Nothing,Float64,Vector{Float64}}
    degree::Int
    kernel::K
    endpoints::E
end

function LocalPolynomial(;
    bandwidth::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    degree::Integer=3,
    kernel::AbstractSmootherKernel=HendersonKernel(),
    endpoints::AbstractEndpoints=MinimumRevision(),
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

(`bandwidth=nothing` means `12Δ` (13 samples) when fitted on its own, and is derived from
the period inside a decomposition.)

The Henderson filter as a smoother: a [`LocalPolynomial`](@ref) of degree 3 with the
[`HendersonKernel`](@ref) (the 13-term filter by default). As a type, `Henderson` is
`LocalPolynomial{HendersonKernel}`, which includes local polynomials of other degrees
with that kernel.
"""
const Henderson = LocalPolynomial{HendersonKernel}

function Henderson(;
    bandwidth::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    endpoints::AbstractEndpoints=MinimumRevision(),
)
    return LocalPolynomial(; bandwidth, degree=3, kernel=HendersonKernel(), endpoints)
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
        "kernel=$(m.kernel), endpoints=$(m.endpoints))",
    )
end

# Fitted on regularly spaced t; predict(f, t_new) evaluates at any time in [t₁, t_N]
function _fit(model::LocalPolynomial, y::AbstractVector{<:Real}, t::AbstractVector{<:Real})
    y, t = _sorted_data(y, t)
    Δ = _require_grid(t, "LocalPolynomial")
    estimated = (;)
    bandwidth = something(model.bandwidth, 2_DEFAULT_HALFWIDTH * Δ)
    if bandwidth isa AbstractVector
        bandwidth, cv_score =
            _select_bandwidth(y, bandwidth, Δ, model.degree, model.kernel)
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

# Bandwidth (twice the radius, in units of t) on a grid of spacing Δ → half-width h in
# samples: the 2h + 1 samples within bandwidth / 2 of x, the outer two included
function _halfwidth(bandwidth::Real, Δ::Real)
    h = bandwidth / 2Δ
    isapprox(h, round(h); rtol=1e-8) && round(h) ≥ 1 || throw(
        ArgumentError(
            "bandwidth / Δt = $(bandwidth / Δ) must be an even integer ≥ 2, got " *
            "bandwidth = $bandwidth",
        ),
    )
    return round(Int, h)
end

# Bandwidth minimizing the mean leave-one-out CV score (P&L §5); the mean, not the sum,
# makes scores comparable across bandwidths with different numbers of interior points
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

# Local polynomial weights on a regular grid, shared by LocalPolynomial and
# MovingAverage. A grid kernel is a weight function κ(d) of the offset d = (tᵢ - x) / Δ in
# samples (not necessarily an integer), with `reach` the largest integer offset of
# positive weight. The weights of the local polynomial fit at offset 0 to observations at
# offsets d are w = KX(XᵀKX)⁻¹e₁ with X = [(d / s)ᵏ], k = 0:degree, scaled by s = reach
# for conditioning (P&L eqs. 3–4). Near the ends of the series, with available offsets dₚ
# and missing future offsets d_f on the grid extended beyond the data, the end weights
# follow the chosen AbstractEndpoints (P&L eqs. 8–9).

struct _GridKernel{F}
    κ::F
    reach::Int
end

function _grid_kernel(kernel::AbstractSmootherKernel, h::Integer)
    return _GridKernel(d -> _kernel_value(kernel, d, h), Int(h))
end

_scale(g::_GridKernel) = max(g.reach, 1)

# Polynomial design [(d / s)ᵏ] for k = 0:degree
function _design(d::AbstractVector, degree::Integer, s::Real)
    return [(dᵢ / s)^k for dᵢ in d, k in 0:degree]
end

_unit(n::Integer) = [1.0; zeros(n - 1)]

function _local_weights(g::_GridKernel, d::AbstractVector, degree::Integer)
    X = _design(d, degree, _scale(g))
    KX = g.κ.(d) .* X
    return KX * ((X' * KX) \ _unit(degree + 1))
end

_end_weights(::NoEndpoints, _, dp, _, _) = fill(NaN, length(dp))
_end_weights(::DirectAsymmetric, g, dp, _, degree) = _local_weights(g, dp, degree)

# Minimum-revision weights (P&L eqs. 8–9 with D = σ²I and a single bias column Z = dᵈ*⁺¹)
# against the target w of the fit on all offsets [dₚ; d_f]. With a = v - wₚ and
# ρ = δ²/σ², the first-order conditions are the KKT system
#     [I + ρZₚZₚᵀ  Uₚ] [a]   [ρZₚ(Z_fᵀw_f)]
#     [Uₚᵀ         0 ] [l] = [U_fᵀw_f     ]
# with Uₚ = [dᵏ], k = 0:d*, on the available offsets (subscript f: the missing ones)
function _end_weights(e::MinimumRevision{Float64}, g, dp, df, degree)
    isinf(e.ratio) &&    # limit ρ → ∞: degree d* + 1 is also reproduced (P&L eq. 12)
        return _end_weights(MinimumRevision(e.degree + 1, 0.0), g, dp, df, degree)
    d = [dp; df]
    p, f = eachindex(dp), (length(dp) + 1):length(d)
    s = _scale(g)
    w = _local_weights(g, d, degree)
    U = _design(d, e.degree, s)
    z = @. (d / s)^(e.degree + 1)
    ρ = e.ratio * Float64(s)^(2 * (e.degree + 1))    # ratio for the scaled column z
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

# The ρ → ∞ limit reproduces degree d* + 1 and must leave a degree of freedom at q = 0
_allows_infinite_ratio(e::MinimumRevision, reach::Integer) = e.degree + 1 < reach

# Validate the end weights and estimate their ratio if it is nothing; returns the
# resolved endpoints and the estimation criteria
function _resolve_endpoints(e::AbstractEndpoints, g::_GridKernel, y, degree)
    _validate(e, g.reach, degree)
    e isa MinimumRevision{Nothing} || return e, (;)
    return _estimate_ratio(e, g, y, degree)
end

const _MIN_MSRE_POINTS = 10

# ρ minimizing the mean square revision error mean((m̂ₜ - m̂ₜ|ₜ)²) of the real-time
# (q = 0) weights over the interior points (P&L §5): grid search over {0, Inf} ∪
# 10^(-4:0.1:2), refined by Brent's method on log₁₀ρ around the best interior grid
# point. Falls back to ρ = 0 when there are too few interior points to estimate it.
function _estimate_ratio(e::MinimumRevision{Nothing}, g::_GridKernel, y, degree)
    r = g.reach
    N = length(y)
    ts = (r + 1):(N - r)
    if length(ts) < _MIN_MSRE_POINTS
        note = "too few points to estimate the ratio by MSRE; using 0"
        return MinimumRevision(e.degree, 0.0), (; ratio=0.0, note_ratio=note)
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
    return MinimumRevision(e.degree, ratio), (; ratio, msre=score)
end

# Fitted smoother on a regular grid: the symmetric weights on offsets -r:r, the end
# weights on -r:q for q = 0:r-1 later observations (mirrored at the start), and the
# fitted values
function _fitted_grid(model, g, endpoints, degree, y, t, Δ, estimated)
    r = g.reach
    N = length(y)
    N ≥ 2r + 1 ||
        throw(ArgumentError("series of length $N is too short for the bandwidth"))
    weights = _local_weights(g, (-r):r, degree)
    end_weights = Vector{Float64}[
        _end_weights(endpoints, g, (-r):q, (q + 1):r, degree) for q in 0:(r - 1)
    ]
    values = fill(NaN, N)
    for i in (r + 1):(N - r)
        values[i] = dot(weights, view(y, (i - r):(i + r)))
    end
    for q in 0:(r - 1)
        v = end_weights[q + 1]
        values[N - q] = dot(v, view(y, (N - q - r):N))
        values[1 + q] = dot(reverse(v), view(y, 1:(1 + q + r)))
    end
    state = (; kernel=g, endpoints, degree, Δ, weights, end_weights)
    return FittedSmoother(model, t, y, values, state, estimated)
end

# Evaluate a grid smoother at arbitrary times in [t₁, t_N]: the fitted values at grid
# points, and the local fit at the continuous offsets elsewhere, with the end weights of
# the grid extended beyond the data where observations are missing on one side
function _predict_grid(f::FittedSmoother, t_new::AbstractVector{<:Real})
    _check_in_range(f.t, t_new)
    (; kernel, endpoints, degree, Δ) = f.state
    N = length(f.y)
    values = Vector{Float64}(undef, length(t_new))
    for (j, x) in enumerate(t_new)
        c = (x - first(f.t)) / Δ + 1    # fractional index of x
        i = round(Int, c)
        values[j] = if isapprox(c, i; atol=1e-9)
            f.values[i]
        else
            _evaluate(kernel, endpoints, degree, f.y, f.values, c, N)
        end
    end
    return values
end

# Near the ends, minimum-revision estimates are interpolated between grid points (cubic
# Lagrange through the four nearest grid estimates): P&L's criterion weights every
# available observation equally, so any off-grid extension jumps wherever an observation
# enters the window, while the interpolant is continuous and exact at grid points
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

# Cubic Lagrange interpolation of the grid values v at the fractional index c
function _lagrange(v, c)
    k = clamp(floor(Int, c), 2, length(v) - 2)
    nodes = (k - 1):(k + 2)
    return sum(nodes) do i
        return v[i] * prod((c - j) / (i - j) for j in nodes if j != i)
    end
end
