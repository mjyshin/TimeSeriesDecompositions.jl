"""
    AbstractSmoother

Supertype of fitted smooth curves: [`LoessSmoother`](@ref). Unlike an
[`AbstractFilter`](@ref), which is a linear operator applied to series on a regular
grid, a smoother is fitted to observations `(tᵢ, yᵢ)` at arbitrary times and evaluated
with `predict(s, t_new)` at any times within the observed range; `predict(s)` returns the
fitted values at the observed times.
"""
abstract type AbstractSmoother end

"""
    LoessSmoother <: AbstractSmoother

Loess curve (Cleveland, 1979; Cleveland & Devlin, 1988): at each time `x`, the weighted
least-squares fit of a polynomial of degree `degree`, evaluated at `x`, with tricube
weights `(1 - (|tᵢ - x| / λ(x))³)³` times the observation weights `wts`. The bandwidth
`λ(x)` is the distance from `x` to its `⌊αN⌋`-th nearest observation for the span `α`
(for `α > 1`, `√α` times the distance to the farthest observation), as in R's `loess`,
or at least the fixed `bandwidth` in units of `t`. It is computed exactly at every point,
matching R's `loess(..., surface = "direct")`.

Fields: `t` (sorted), `y`, `wts`, `span`, `bandwidth`, `degree`, `fitted` (at `t`) and
`estimated` (the hyperparameters chosen from the data, with the criteria that chose
them). Create with `fit(LoessSmoother, y, t)`.
"""
struct LoessSmoother <: AbstractSmoother
    t::Vector{Float64}
    y::Vector{Float64}
    wts::Vector{Float64}
    span::Union{Nothing,Float64}
    bandwidth::Union{Nothing,Float64}
    degree::Int
    fitted::Vector{Float64}
    estimated::NamedTuple
end

function Base.show(io::IO, s::LoessSmoother)
    print(io, "LoessSmoother(")
    isnothing(s.span) || print(io, "span=", round(s.span; sigdigits=4), ", ")
    isnothing(s.bandwidth) || print(io, "bandwidth=", round(s.bandwidth; sigdigits=4), ", ")
    print(io, "degree=$(s.degree))")
    isempty(s.estimated) || print(io, " [estimated: ", join(keys(s.estimated), ", "), "]")
    return nothing
end

"""
    fit(LoessSmoother, y, t=eachindex(y); span=nothing, bandwidth=nothing, degree=2,
        wts=nothing) -> LoessSmoother

Fit a loess curve to observations `y` at times `t` (any order and spacing).

# Keyword arguments
- `span`: the fraction `α` of the observations in each local fit, as in R's `loess`:
  the bandwidth at `x` is the distance to its `⌊αN⌋`-th nearest observation, so it
  adapts to the local density of `t`. `α > 1` gives smoother fits, with the distance to
  the farthest observation times `√α`, as R's `loess` computes.
- `bandwidth`: a bandwidth in units of `t`. Alone, it gives a fixed-bandwidth smoother,
  which smooths over the same time scale however irregular `t` is (every local fit must
  still contain `degree + 1` observations). With `span`, the bandwidth at `x` is the
  larger of the two, as in `locfit`, so that gaps are bridged.
- `degree`: degree of the local polynomials, `0`, `1` or `2` (default `2`).
- `wts`: non-negative observation weights (default all ones), e.g. the robustness
  weights of STL.

If neither `span` nor `bandwidth` is given, the span is chosen by minimising the AICc of
Hurvich, Simonoff & Tsai (1998), `log σ̂² + 1 + 2(tr L + 1) / (N - tr L - 2)` for the hat
matrix `L`, which is reported with the equivalent number of parameters `tr L`. The AICc
assumes independent errors and undersmooths when they are positively autocorrelated, in
which case correlated noise and a wiggly trend cannot be told apart from one series; a
warning (log group `:diagnostics`) is issued when the lag-one autocorrelation of the
residuals exceeds its expectation under independent errors by more than three standard
deviations. The check is skipped, with a note, when the chosen span is so large that its
hat matrix would exceed two million entries. Choosing the span costs `O(N²)` time
(about a second for `N = 3000`); pass `span` or `bandwidth` for long series.

Hyperparameters chosen from the data are reported with an `@info` message and stored in
the field `estimated`.
"""
function fit(
    ::Type{LoessSmoother},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    span::Union{Nothing,Real}=nothing,
    bandwidth::Union{Nothing,Real}=nothing,
    degree::Integer=2,
    wts::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    s = _fit(LoessSmoother, y, t; span, bandwidth, degree, wts)
    _log_estimated(LoessSmoother, s.estimated)
    return s
end

function _fit(
    ::Type{LoessSmoother},
    y::AbstractVector{<:Real},
    t::AbstractVector{<:Real}=eachindex(y);
    span::Union{Nothing,Real}=nothing,
    bandwidth::Union{Nothing,Real}=nothing,
    degree::Integer=2,
    wts::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    y = _check_series(y)
    N = length(y)
    length(t) == N || throw(DimensionMismatch("y and t must have the same length"))
    all(isfinite, t) || throw(ArgumentError("t must be finite"))
    degree ∈ (0, 1, 2) || throw(DomainError(degree, "degree must be 0, 1 or 2"))
    isnothing(span) || span > 0 || throw(DomainError(span, "span must be positive"))
    isnothing(bandwidth) ||
        bandwidth > 0 ||
        throw(DomainError(bandwidth, "bandwidth must be positive"))
    wts = something(wts, ones(N))
    length(wts) == N || throw(DimensionMismatch("wts and y must have the same length"))
    all(w -> isfinite(w) && w ≥ 0, wts) ||
        throw(ArgumentError("wts must be finite and non-negative"))
    perm = sortperm(t)
    t, y, wts = Float64.(t[perm]), y[perm], Float64.(wts[perm])

    estimated = (;)
    if isnothing(span) && isnothing(bandwidth)
        q, criteria = _select_span(t, y, wts, degree)
        span = q / N
        estimated = (; span, criteria...)
    end
    isnothing(span) ||
        _span_neighbours(span, N) ≥ degree + 2 ||
        throw(ArgumentError("span × N must be at least degree + 2 = $(degree + 2)"))
    α, h = something(span, 0.0), something(bandwidth, 0.0)
    fitted = first(_loess_fit(t, y, wts, α, h, degree))
    return LoessSmoother(
        t, y, wts, _float(span), _float(bandwidth), Int(degree), fitted, estimated
    )
end

_float(x) = isnothing(x) ? nothing : Float64(x)

# Number of nearest neighbours q = ⌊αN + 10⁻⁵⌋ for the span α, the rounding of R's loess
_span_neighbours(span, N) = floor(Int, span * N + 1e-5)

"""
    predict(s::LoessSmoother) -> Vector{Float64}
    predict(s::LoessSmoother, t_new; extrapolate=false) -> Vector{Float64}

Fitted values at the observed times `s.t`, or the loess curve evaluated exactly at the
times `t_new`, which must lie within `[s.t[1], s.t[end]]` unless `extrapolate=true`
(beyond the data, the local polynomial of the nearest observations is continued).
"""
predict(s::LoessSmoother) = s.fitted

function predict(s::LoessSmoother, t_new::AbstractVector{<:Real}; extrapolate::Bool=false)
    t₁, t_N = first(s.t), last(s.t)
    if !extrapolate
        i = findfirst(x -> !(t₁ ≤ x ≤ t_N), t_new)
        isnothing(i) || throw(
            DomainError(t_new[i], "t_new must lie within [$t₁, $t_N]; see extrapolate")
        )
    end
    α, h = something(s.span, 0.0), something(s.bandwidth, 0.0)
    return [_local_value(s.t, s.y, s.wts, α, h, s.degree, Float64(x)) for x in t_new]
end

fitted(s::LoessSmoother) = s.fitted
residuals(s::LoessSmoother) = s.y - s.fitted

# Local fit at x: the observations idx within the bandwidth λ of x (contiguous in the
# sorted t), λ, and the coefficients c of the equivalent kernel, so that observation i
# has weight lᵢ = wᵢ c(uᵢ) with uᵢ = (tᵢ - x) / λ, the polynomial c(u) = Σₖ cₖ uᵏ and
# wᵢ the tricube times observation weight: c = G⁻¹e₁ for the moments Gⱼₖ = Σᵢ wᵢ uᵢʲ⁺ᵏ.
# λ is the larger of the fixed bandwidth h and, for the span α > 0, the distance to the
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
# observations within the bandwidth of tᵢ)
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
        "LoessSmoother: the residuals are more autocorrelated than independent errors " *
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
