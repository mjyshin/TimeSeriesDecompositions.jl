"""
    LocalPolynomial(; bandwidth=nothing, degree=3, kernel=HendersonKernel(),
                    endpoints=MinimumRevision()) <: AbstractTrendSmoother

Local polynomial trend filter [1] for regularly spaced `t` of spacing `Δ`: at each time, the
`kernel`-weighted least-squares polynomial of degree `degree` through the observations
within `bandwidth / 2`, evaluated there. At the observed times it is the linear filter of
X-11; between them, the same fit at the continuous offsets.

- `bandwidth`: window width `2hΔ` in units of `t` (`h ≥ 1` an integer), or a vector of
  candidates, of which the one minimizing the leave-one-out cross-validation score is
  chosen. `nothing` chooses among `2hΔ` up to `h = (N - 1) ÷ 4` (inside a decomposition,
  from the period up).
- `degree`: degree of the local polynomial.
- `kernel`: an [`AbstractSmootherKernel`](@ref).
- `endpoints`: an [`AbstractEndpoints`](@ref) for the first and last `h` samples.

The leave-one-out residual of a linear smoother is `(yᵢ - ŷᵢ) / (1 - wᵢᵢ)`, so no refit is
needed. Candidates are scored over all `N` points, each with its own end weights (a
minimum-revision `ratio` left as `nothing` is estimated for each), or with
[`NoEndpoints`](@ref) over the interior of the widest. A warning (log group `:diagnostics`)
is issued when the chosen filter's residuals are more autocorrelated than independent
errors allow, which makes the choice too short.

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
    bandwidth = something(
        model.bandwidth, _bandwidth_grid(_min_halfwidth(model.degree), length(y), Δ))
    if bandwidth isa AbstractVector
        bandwidth, criteria = _select_bandwidth(
            y, bandwidth, Δ, model.degree, model.kernel, model.endpoints)
        estimated = (; bandwidth, criteria...)
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

# Smallest half-width of a local polynomial of degree d (2h > d), and candidate bandwidths
# from half-width h_min up to (N - 1) ÷ 4
_min_halfwidth(degree) = max(1, cld(degree + 1, 2))

function _bandwidth_grid(h_min, N, Δ)
    h_max = max(h_min, (N - 1) ÷ 4)
    return collect(2.0 .* (h_min:h_max) .* Δ)
end

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

# Bandwidth minimizing the mean leave-one-out score (P&L §5) of the filter with its end
# weights over all N points, so every candidate is scored on the same points; NoEndpoints
# leaves the ends undefined, so it is scored over the interior of the widest candidate.
# Then the residual autocorrelation check of the chosen filter's interior
function _select_bandwidth(y, bandwidths, Δ, degree, kernel, endpoints)
    isempty(bandwidths) && throw(ArgumentError("no candidate bandwidths"))
    N = length(y)
    halfwidths = [_halfwidth(b, Δ) for b in bandwidths]
    valid = [N ≥ 2h + 1 && 2h > degree for h in halfwidths]
    any(valid) || throw(ArgumentError("no candidate bandwidth fits the series"))
    H = maximum(halfwidths[valid])
    ts = endpoints isa NoEndpoints ? ((H + 1):(N - H)) : (1:N)
    scores = map(halfwidths, valid) do h, ok
        ok || return Inf
        g = _grid_kernel(kernel, h)
        e = try
            first(_resolve_endpoints(endpoints, g, y, degree))
        catch err
            err isa ArgumentError || rethrow()
            return Inf    # end weights undefined at this bandwidth
        end
        values, self = _filter_values(_filter_weights(g, e, degree)..., y)
        all(<(1 - 1e-8), view(self, ts)) || return Inf
        return mean(t -> ((y[t] - values[t]) / (1 - self[t]))^2, ts)
    end
    i = argmin(scores)
    isfinite(scores[i]) ||
        throw(ArgumentError("no candidate bandwidth gives a finite cross-validation score"))
    h = halfwidths[i]
    w = _local_weights(_grid_kernel(kernel, h), (-h):h, degree)
    z = _residual_autocorrelation_z(_interior_residual_matrix(w, N), y)
    _warn_autocorrelation(:LocalPolynomial, z, "bandwidth")
    return bandwidths[i], (; cv_score = scores[i], autocorrelation_z = z)
end

# Map from y to the residuals y_t - w ⋅ y[t-h:t+h] of a symmetric filter w at interior t
function _interior_residual_matrix(w, N)
    h = length(w) ÷ 2
    rows, cols, vals = Int[], Int[], Float64[]
    for (r, t) in enumerate((h + 1):(N - h)), (k, j) in enumerate((t - h):(t + h))

        push!(rows, r)
        push!(cols, j)
        push!(vals, (j == t) - w[k])
    end
    return sparse(rows, cols, vals, N - 2h, N)
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
# P&L eqs. 8–9 with D = σ²I and bias column Z = dᵈ*⁺¹,
#     [M Uₚ; Uₚᵀ 0] [x; λ] = [b₁; b₂],  M = I + ρzₚzₚᵀ,
# solved with M⁻¹ by Sherman–Morrison and the k × k Schur complement UₚᵀM⁻¹Uₚ, in O(nk²)
function _end_weights(e::MinimumRevision{Float64}, g, dp, df, degree)
    isinf(e.ratio) &&    # P&L eq. 12
        return _end_weights(MinimumRevision(e.degree + 1, 0.0), g, dp, df, degree)
    return _min_revision_weights(_min_revision_system(e.degree, g, dp, df, degree), e.ratio)
end

# The parts of the system that do not depend on the ratio
function _min_revision_system(bias_degree, g, dp, df, degree)
    d = [dp; df]
    p, f = eachindex(dp), (length(dp) + 1):length(d)
    s = _scale(g)
    w = _local_weights(g, d, degree)
    U = _design(d, bias_degree, s)
    z = @. (d / s)^(bias_degree + 1)
    scale = Float64(s)^(2 * (bias_degree + 1))    # of the ratio, for the scaled column z
    return (; wp = w[p], Up = U[p, :], zp = z[p], zf_wf = dot(view(z, f), view(w, f)),
        b₂ = U[f, :]' * view(w, f), scale)
end

function _min_revision_weights(sys, ratio::Real)
    (; wp, Up, zp, zf_wf, b₂, scale) = sys
    ρ = ratio * scale
    b₁ = (ρ * zf_wf) .* zp
    c = ρ / (1 + ρ * dot(zp, zp))
    M⁻¹Up = Up .- (c .* zp) * (zp' * Up)
    M⁻¹b₁ = b₁ .- (c * dot(zp, b₁)) .* zp
    λ = (Up' * M⁻¹Up) \ (Up' * M⁻¹b₁ .- b₂)
    return wp .+ M⁻¹b₁ .- M⁻¹Up * λ
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
    windows = [y[t + j] for t in ts, j in (-r):0]    # the real-time windows, one per row
    sys = _min_revision_system(e.degree, g, (-r):0, 1:r, degree)
    sys_inf = _allows_infinite_ratio(e, r) ?
              _min_revision_system(e.degree + 1, g, (-r):0, 1:r, degree) : nothing
    function msre(ρ)
        v = isinf(ρ) ? _min_revision_weights(sys_inf, 0.0) : _min_revision_weights(sys, ρ)
        return mean(abs2, final .- windows * v)
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
    N = length(y)
    N ≥ 2g.reach + 1 ||
        throw(ArgumentError("series of length $N is too short for the bandwidth"))
    weights, end_weights = _filter_weights(g, endpoints, degree)
    values, _ = _filter_values(weights, end_weights, y)
    state = (; kernel = g, endpoints, degree, Δ, weights, end_weights)
    return FittedSmoother(model, t, y, values, state, estimated)
end

# Weights at offsets -r:r, and for q = 0:r-1 the end weights at offsets -r:q
function _filter_weights(g::_GridKernel, endpoints, degree)
    r = g.reach
    weights = _local_weights(g, (-r):r, degree)
    end_weights = Vector{Float64}[_end_weights(endpoints, g, (-r):q, (q + 1):r, degree)
                                  for q in 0:(r - 1)]
    return weights, end_weights
end

# The filter applied to y (end weights mirrored at the start), and the weight of each value
# on its own observation
function _filter_values(weights, end_weights, y)
    r = length(weights) ÷ 2
    N = length(y)
    values, self = fill(NaN, N), fill(NaN, N)
    for i in (r + 1):(N - r)
        values[i] = dot(weights, view(y, (i - r):(i + r)))
        self[i] = weights[r + 1]
    end
    for q in 0:(r - 1)
        v = end_weights[q + 1]
        values[N - q] = dot(v, view(y, (N - q - r):N))
        values[1 + q] = dot(reverse(v), view(y, 1:(1 + q + r)))
        self[N - q] = self[1 + q] = v[r + 1]
    end
    return values, self
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
