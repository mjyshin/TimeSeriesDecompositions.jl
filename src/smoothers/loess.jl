"""
    LOESS(; span=nothing, bandwidth=nothing, degree=2) <: AbstractTrendSmoother

Loess (Cleveland, 1979; Cleveland & Devlin, 1988) for any times: at each time `x`, the
weighted least-squares fit of a polynomial of degree `degree`, evaluated at `x`, with
tricube weights `(1 - (|tᵢ - x| / λ(x))³)³` times the observation weights. It is computed
exactly at every point, matching R's `loess(..., surface = "direct")`.

- `span`: the fraction `α` of the observations in each local fit, as in R's `loess`:
  the radius `λ(x)` is the distance to the `⌊αN⌋`-th nearest observation, so it
  adapts to the local density of `t`. `α > 1` gives smoother fits, with `λ` the distance
  to the farthest observation times `√α`, as R's `loess` computes.
- `bandwidth`: the width `2λ` of the local fits in units of `t`, twice the radius, so
  that on a grid of spacing `Δ`, `bandwidth = 2hΔ` reaches the `2h + 1` samples at
  offsets `-h:h` (the outer two with tricube weight zero). Alone, it gives a
  fixed-bandwidth smoother, which smooths over the same time scale however irregular `t`
  is (every local fit must still contain `degree + 1` observations with positive
  weight). With `span`, the radius at `x` is the larger of the two, as in `locfit`, so
  that gaps are bridged.
- `degree`: degree of the local polynomials, `0`, `1` or `2`.

If neither `span` nor `bandwidth` is given, the span is chosen by minimising the AICc of
Hurvich, Simonoff & Tsai (1998), `log σ̂² + 1 + 2(tr L + 1) / (N - tr L - 2)` for the hat
matrix `L`, which is reported with the equivalent number of parameters `tr L` (inside a
decomposition, `nothing` instead derives the span from the period). The AICc assumes
independent errors and undersmooths when they are positively autocorrelated, in which
case correlated noise and a wiggly trend cannot be told apart from one series; a warning
(log group `:diagnostics`) is issued when the lag-one autocorrelation of the residuals
exceeds its expectation under independent errors by more than three standard deviations.
The check is skipped, with a note, when the chosen span is so large that its hat matrix
would exceed two million entries. Choosing the span costs `O(N²)` time (about a second
for `N = 3000`); pass `span` or `bandwidth` for long series.

`fit(model, y, t; wts=nothing)` takes optional non-negative observation weights `wts`
(e.g. the robustness weights of STL). `predict(f, t_new; extrapolate=false)` evaluates
the curve exactly at times within `[t₁, t_N]`, or beyond them with `extrapolate=true`
(continuing the local polynomial of the nearest observations).
"""
struct LOESS <: AbstractTrendSmoother
    span::Union{Nothing,Float64}
    bandwidth::Union{Nothing,Float64}
    degree::Int
end

function LOESS(;
    span::Union{Nothing,Real}=nothing,
    bandwidth::Union{Nothing,Real}=nothing,
    degree::Integer=2,
)
    isnothing(span) || span > 0 || throw(DomainError(span, "span must be positive"))
    isnothing(bandwidth) ||
        bandwidth > 0 ||
        throw(DomainError(bandwidth, "bandwidth must be positive"))
    degree ∈ (0, 1, 2) || throw(DomainError(degree, "degree must be 0, 1 or 2"))
    return LOESS(_float(span), _float(bandwidth), Int(degree))
end

function Base.show(io::IO, m::LOESS)
    print(io, "LOESS(")
    isnothing(m.span) || print(io, "span=", _fmt(m.span), ", ")
    isnothing(m.bandwidth) || print(io, "bandwidth=", _fmt(m.bandwidth), ", ")
    return print(io, "degree=$(m.degree))")
end

function _fit(
    model::LOESS,
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real};
    wts::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    N = length(y)
    wts = something(wts, ones(N))
    length(wts) == N || throw(DimensionMismatch("wts and y must have the same length"))
    all(w -> isfinite(w) && w ≥ 0, wts) ||
        throw(ArgumentError("wts must be finite and non-negative"))
    perm = sortperm(t)
    y, t = _sorted_data(y, t)
    wts = Float64.(wts[perm])
    (; span, bandwidth, degree) = model

    estimated = (;)
    if isnothing(span) && isnothing(bandwidth)
        q, criteria = _select_span(t, y, wts, degree)
        span = q / N
        estimated = (; span, criteria...)
    end
    isnothing(span) ||
        _span_neighbours(span, N) ≥ degree + 2 ||
        throw(ArgumentError("span × N must be at least degree + 2 = $(degree + 2)"))
    α, h = something(span, 0.0), _radius(bandwidth)
    values = first(_loess_fit(t, y, wts, α, h, degree))
    resolved = LOESS(_float(span), bandwidth, degree)
    return FittedSmoother(resolved, t, y, values, (; wts), estimated)
end

_float(x) = isnothing(x) ? nothing : Float64(x)

# Fixed radius λ of a bandwidth (twice the radius), 0 for none
_radius(bandwidth) = isnothing(bandwidth) ? 0.0 : bandwidth / 2

# Number of nearest neighbours q = ⌊αN + 10⁻⁵⌋ for the span α, the rounding of R's loess
_span_neighbours(span, N) = floor(Int, span * N + 1e-5)

function predict(
    f::FittedSmoother{LOESS}, t_new::AbstractVector{<:Real}; extrapolate::Bool=false
)
    extrapolate || _check_in_range(f.t, t_new)
    α, h = something(f.model.span, 0.0), _radius(f.model.bandwidth)
    return [
        _local_value(f.t, f.y, f.state.wts, α, h, f.model.degree, Float64(x)) for x in t_new
    ]
end

# Local fit at x: the observations idx within the radius λ of x (contiguous in the
# sorted t), λ, and the coefficients c of the equivalent kernel, so that observation i
# has weight lᵢ = wᵢ c(uᵢ) with uᵢ = (tᵢ - x) / λ, the polynomial c(u) = Σₖ cₖ uᵏ and
# wᵢ the tricube times observation weight: c = G⁻¹e₁ for the moments Gⱼₖ = Σᵢ wᵢ uᵢʲ⁺ᵏ.
# λ is the larger of the fixed radius h and, for the span α > 0, the distance to the
# ⌊αN⌋-th nearest observation; for α > 1, the distance to the farthest one times √α,
# as R's loess computes for one predictor (its documentation states α^(1/p))
function _local_fit(t::Vector{Float64}, wts::Vector{Float64}, α, h, degree, x::Float64)
    N = length(t)
    q = α > 0 ? _span_neighbours(α, N) : 0
    λ = h
    if q > 0
        hi = searchsortedfirst(t, x) - 1    # [lo, hi] starts empty between t[hi] and t[lo]
        lo = hi + 1
        for _ in 1:min(q, N)    # add the nearer of the two neighbouring observations
            if hi == N || (lo > 1 && x - t[lo - 1] ≤ t[hi + 1] - x)
                lo -= 1
            else
                hi += 1
            end
        end
        λ = max(λ, max(x - t[lo], t[hi] - x) * sqrt(max(1, α)))
    end
    λ > 0 || throw(
        ArgumentError("$q or more observations are tied at t = $x; increase the span")
    )
    idx = searchsortedfirst(t, x - λ):searchsortedlast(t, x + λ)
    moments = zeros(2degree + 1)
    distinct, previous = 0, NaN
    for i in idx
        u = (t[i] - x) / λ
        w = _tricube_weight(wts[i], u)
        w > 0 || continue
        t[i] == previous || (distinct += 1)
        previous = t[i]
        for k in eachindex(moments)
            moments[k] += w * u^(k - 1)
        end
    end
    distinct > degree || throw(
        ArgumentError(
            "fewer than $(degree + 1) distinct observations with positive weight near " *
            "t = $x; increase the span or bandwidth",
        ),
    )
    G = [moments[j + k - 1] for j in 1:(degree + 1), k in 1:(degree + 1)]
    return idx, λ, G \ _unit(degree + 1)
end

_tricube_weight(wᵢ, u) = abs(u) < 1 ? wᵢ * (1 - abs(u)^3)^3 : 0.0

# Weight of observation i in the local fit at x
function _local_weight(t, wts, x, λ, c, i)
    u = (t[i] - x) / λ
    return _tricube_weight(wts[i], u) * evalpoly(u, c)
end

# Local fit at x applied to y
function _local_value(t, y, wts, α, h, degree, x)
    idx, λ, c = _local_fit(t, wts, α, h, degree, x)
    return sum(_local_weight(t, wts, x, λ, c, i) * y[i] for i in idx)
end

# Fitted values at the observed times and the diagonal of the hat matrix L (at x = tᵢ,
# uᵢ = 0, so Lᵢᵢ = wtsᵢ c₀)
function _loess_fit(t, y, wts, α, h, degree)
    N = length(t)
    fitted, leverage = zeros(N), zeros(N)
    for i in 1:N
        idx, λ, c = _local_fit(t, wts, α, h, degree, t[i])
        fitted[i] = sum(_local_weight(t, wts, t[i], λ, c, j) * y[j] for j in idx)
        leverage[i] = wts[i] * c[1]
    end
    return fitted, leverage
end

# Hat matrix L of the loess fit at the observed times (banded: row i spans the
# observations within the radius of tᵢ)
function _hat_matrix(t, wts, α, h, degree)
    N = length(t)
    rows, cols, vals = Int[], Int[], Float64[]
    for i in 1:N
        idx, λ, c = _local_fit(t, wts, α, h, degree, t[i])
        append!(rows, fill(i, length(idx)))
        append!(cols, idx)
        append!(vals, (_local_weight(t, wts, t[i], λ, c, j) for j in idx))
    end
    return sparse(rows, cols, vals, N, N)
end

# Largest number of hat-matrix entries N·q for which the autocorrelation check is run
const _MAX_DIAGNOSTIC_ENTRIES = 2_000_000

# Number of neighbours q minimising the AICc log σ̂² + 1 + 2(tr L + 1) / (n - tr L - 2)
# over about 40 values spaced geometrically between degree + 2 and N, skipping those
# that leave some local fit with too few distinct observations; σ̂² is weighted by the
# observation weights and n is the number of observations with positive weight. The residuals of
# the chosen fit are then checked for autocorrelation, unless its hat matrix is too
# large; undersmoothing, which the check guards against, gives small spans.
function _select_span(t, y, wts, degree)
    N = length(t)
    N ≥ degree + 5 || throw(ArgumentError("too few observations to choose the span"))
    candidates = unique(round.(Int, exp.(range(log(degree + 2), log(N); length=40))))
    best = (q=0, aicc=Inf, enp=NaN)
    n = count(>(0), wts)    # observations with positive weight
    for q in candidates
        fitted, leverage = try
            _loess_fit(t, y, wts, q / N, 0.0, degree)
        catch err
            err isa ArgumentError || rethrow()
            continue    # too few distinct observations at some t, e.g. with ties
        end
        enp = sum(leverage)
        n - enp - 2 > 0 || continue
        σ² = sum(wts .* (y - fitted) .^ 2) / sum(wts)
        aicc = log(σ²) + 1 + 2(enp + 1) / (n - enp - 2)
        aicc < best.aicc && (best = (; q, aicc, enp))
    end
    isfinite(best.aicc) || throw(ArgumentError("no span gives a finite AICc"))
    criteria = (; enp=best.enp, aicc=best.aicc)
    if N * best.q > _MAX_DIAGNOSTIC_ENTRIES
        note = "residual autocorrelation not checked: N × q exceeds $_MAX_DIAGNOSTIC_ENTRIES"
        return best.q, (; criteria..., autocorrelation_note=note)
    end
    z = _residual_autocorrelation_z(_hat_matrix(t, wts, best.q / N, 0.0, degree), y)
    z > 3 && @warn(
        "LOESS: the residuals are more autocorrelated than independent errors " *
            "would give (z = $(round(z; digits=1))); the AICc span is likely too small",
        _group = :diagnostics,
    )
    return best.q, (; criteria..., autocorrelation_z=z)
end

# Standardised excess lag-one autocorrelation of the residuals r = My, M = I - L, over
# its expectation under independent errors: with the lag-one shift S, A = Mᵀ(S + Sᵀ)M/2
# and B = MᵀM, ρ̂ = yᵀAy / yᵀBy has approximate mean ρ₀ = tr A / tr B and standard
# deviation √(2 tr C²) / tr B for C = A - ρ₀B
function _residual_autocorrelation_z(L, y)
    N = length(y)
    M = I - L
    S = spdiagm(1 => ones(N - 1))
    A = M' * (S + S') * M / 2
    B = M' * M
    ρ₀ = tr(A) / tr(B)
    C = A - ρ₀ * B
    sd = sqrt(2 * sum(abs2, C)) / tr(B)    # tr C² = ‖C‖²_F for symmetric C
    return (dot(y, A * y) / dot(y, B * y) - ρ₀) / sd
end
