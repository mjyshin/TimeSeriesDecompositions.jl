"""
    LOESS(; span=nothing, bandwidth=nothing, degree=2) <: AbstractTrendSmoother

Loess [1, 2] at any times: at each time, the tricube-weighted (times observation weights)
least-squares polynomial of degree `degree`, computed exactly as R's
`loess(..., surface = "direct")`.

- `span`: fraction `α` of the observations in each local fit, as in R's `loess`, or a
  vector of candidates, of which the one minimizing the leave-one-out cross-validation
  score is chosen.
- `bandwidth`: fixed window width `2λ` in units of `t`; with `span`, the radius is the
  larger of the two.
- `degree`: `0`, `1` or `2`.

With neither `span` nor `bandwidth`, the span is chosen among geometrically spaced numbers
of neighbors (inside a decomposition, from the period up). The leave-one-out residual is
`(yᵢ - ŷᵢ) / (1 - Lᵢᵢ)` for the leverage `Lᵢᵢ` with the window held fixed, in `O(N²)` time
in all. A warning (log group `:diagnostics`) is issued when the residuals' lag-one
autocorrelation exceeds its expectation under independent errors by more than three
standard deviations, which makes the chosen span too small.

`fit(model, y, t; wts=nothing)` takes non-negative observation weights.
`predict(f, t_new; extrapolate=false)` evaluates within `[t₁, t_N]`, or beyond with
`extrapolate=true`. Unlike Loess.jl, it supports observation weights, fixed bandwidths and
exact evaluation.

[1] W. S. Cleveland. "Robust locally weighted regression and smoothing scatterplots". JASA
74 (368), 1979.

[2] W. S. Cleveland and S. J. Devlin. "Locally weighted regression: an approach to
regression analysis by local fitting". JASA 83 (403), 1988.
"""
struct LOESS <: AbstractTrendSmoother
    span::Union{Nothing, Float64, Vector{Float64}}
    bandwidth::Union{Nothing, Float64}
    degree::Int
end

function LOESS(;
        span::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        bandwidth::Union{Nothing, Real} = nothing,
        degree::Integer = 2
)
    isnothing(span) || all(>(0), span) || throw(DomainError(span, "span must be positive"))
    span = span isa AbstractVector ? collect(Float64, span) : _float(span)
    isnothing(bandwidth) ||
        bandwidth > 0 ||
        throw(DomainError(bandwidth, "bandwidth must be positive"))
    degree ∈ (0, 1, 2) || throw(DomainError(degree, "degree must be 0, 1 or 2"))
    return LOESS(span, _float(bandwidth), Int(degree))
end

function Base.show(io::IO, m::LOESS)
    print(io, "LOESS(")
    isnothing(m.span) || print(io, "span=", _fmt_bandwidth(m.span), ", ")
    isnothing(m.bandwidth) || print(io, "bandwidth=", _fmt(m.bandwidth), ", ")
    return print(io, "degree=$(m.degree))")
end

function _fit(
        model::LOESS,
        y::AbstractVector{<:Real},
        t::AbstractVector{<:Real};
        wts::Union{Nothing, AbstractVector{<:Real}} = nothing
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
    if span isa AbstractVector || isnothing(span) && isnothing(bandwidth)
        candidates = isnothing(span) ? _neighbor_grid(degree + 2, N) :
                     unique(_span_neighbors.(span, N))
        q, criteria = _select_span(t, y, wts, degree, candidates, _radius(bandwidth))
        span = q / N
        estimated = (; span, criteria...)
    end
    isnothing(span) ||
        _span_neighbors(span, N) ≥ degree + 2 ||
        throw(ArgumentError("span × N must be at least degree + 2 = $(degree + 2)"))
    α, h = something(span, 0.0)::Float64, _radius(bandwidth)
    values = first(_loess_fit(t, y, wts, α, h, degree))
    resolved = LOESS(_float(span), bandwidth, degree)
    return FittedSmoother(resolved, t, y, values, (; wts), estimated)
end

_float(x) = isnothing(x) ? nothing : Float64(x)

_radius(bandwidth) = isnothing(bandwidth) ? 0.0 : bandwidth / 2

# Neighbors q = ⌊αN + 10⁻⁵⌋ of the span α, rounded as R's loess
_span_neighbors(span, N) = floor(Int, span * N + 1e-5)

function predict(
        f::FittedSmoother{LOESS}, t_new::AbstractVector{<:Real}; extrapolate::Bool = false
)
    extrapolate || _check_in_range(f.t, t_new)
    α, h = something(f.model.span, 0.0)::Float64, _radius(f.model.bandwidth)
    return [_local_value(f.t, f.y, f.state.wts, α, h, f.model.degree, Float64(x))
            for x in t_new]
end

# Local fit at x: the observations idx within the radius λ and the coefficients c of the
# equivalent kernel, so that observation i has weight wᵢ c(uᵢ), uᵢ = (tᵢ - x) / λ, for the
# tricube-times-observation weight wᵢ; c = G⁻¹e₁ for the moments Gⱼₖ = Σᵢ wᵢ uᵢʲ⁺ᵏ.
# For α > 1, R's loess scales the radius by √α for one predictor (its documentation says α^(1/p)).
function _local_fit(t::Vector{Float64}, wts::Vector{Float64}, α, h, degree, x::Float64)
    N = length(t)
    q = α > 0 ? _span_neighbors(α, N) : 0
    λ = h
    if q > 0
        hi = searchsortedfirst(t, x) - 1    # [lo, hi] starts empty
        lo = hi + 1
        for _ in 1:min(q, N)
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

function _local_weight(t, wts, x, λ, c, i)
    u = (t[i] - x) / λ
    return _tricube_weight(wts[i], u) * evalpoly(u, c)
end

function _local_value(t, y, wts, α, h, degree, x)
    idx, λ, c = _local_fit(t, wts, α, h, degree, x)
    return sum(_local_weight(t, wts, x, λ, c, i) * y[i] for i in idx)
end

# Fitted values and the diagonal of the hat matrix, Lᵢᵢ = wtsᵢ c₀
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

const _MAX_DIAGNOSTIC_ENTRIES = 2_000_000    # of the hat matrix, for the autocorrelation check

# Numbers of neighbors from q_min to N, about 5% apart
function _neighbor_grid(q_min, N)
    qs = Int[]
    q = min(q_min, N)
    while q < N
        push!(qs, q)
        q = max(q + 1, round(Int, 1.05q))
    end
    return push!(qs, N)
end

# Neighbors q minimizing the leave-one-out score, then a check of the residual
# autocorrelation
function _select_span(t, y, wts, degree, candidates, h)
    N = length(t)
    best = (q = 0, cv_score = Inf, enp = NaN)
    for q in candidates
        q ≥ degree + 2 || continue
        fitted, leverage = try
            _loess_fit(t, y, wts, q / N, h, degree)
        catch err
            err isa ArgumentError || rethrow()
            continue    # too few distinct observations at some t
        end
        keep = wts .> 0
        all(<(1 - 1e-8), leverage[keep]) || continue
        score = sum(wts[keep] .* ((y - fitted) ./ (1 .- leverage)) .^ 2 ./ sum(wts[keep]))
        score < best.cv_score && (best = (; q, cv_score = score, enp = sum(leverage)))
    end
    isfinite(best.cv_score) ||
        throw(ArgumentError("no candidate span gives a finite cross-validation score"))
    criteria = (; best.cv_score, best.enp)
    if N * best.q > _MAX_DIAGNOSTIC_ENTRIES
        note = "residual autocorrelation not checked: N × q exceeds $_MAX_DIAGNOSTIC_ENTRIES"
        return best.q, (; criteria..., note_autocorr = note)
    end
    z = _residual_autocorrelation_z(I - _hat_matrix(t, wts, best.q / N, h, degree), y)
    _warn_autocorrelation(:LOESS, z, "span")
    return best.q, (; criteria..., autocorrelation_z = z)
end

function _warn_autocorrelation(name, z, hyperparameter)
    z > 3 &&
        @warn("$name: the residuals are more autocorrelated than independent errors " *
              "would give (z = $(round(z; digits=1))); the chosen $hyperparameter is " *
              "likely too small",
            _group = :diagnostics)
    return nothing
end

# z-score of the lag-one autocorrelation ρ̂ = yᵀAy / yᵀBy of the residuals My against
# independent errors, A = Mᵀ(S + Sᵀ)M/2 and B = MᵀM for the lag-one shift S
function _residual_autocorrelation_z(M, y)
    S = spdiagm(1 => ones(size(M, 1) - 1))
    A = M' * (S + S') * M / 2
    B = M' * M
    ρ₀ = tr(A) / tr(B)
    C = A - ρ₀ * B
    sd = sqrt(2 * sum(abs2, C)) / tr(B)    # tr C² = ‖C‖²_F for symmetric C
    return (dot(y, A * y) / dot(y, B * y) - ρ₀) / sd
end
