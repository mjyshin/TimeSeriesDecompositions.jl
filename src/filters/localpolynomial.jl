"""
    AbstractEndpoints

Scheme for the asymmetric weights of a [`LocalPolynomialFilter`](@ref) at the last (and,
by mirror symmetry, first) `h` points of a series, where the symmetric filter of length
`2h + 1` does not fit: [`DirectAsymmetric`](@ref) or [`MinimumRevision`](@ref). Passing
`endpoints=nothing` instead leaves those points as `NaN`.
"""
abstract type AbstractEndpoints end

"""
    DirectAsymmetric()

Direct asymmetric filter (DAF): at the end of the series, refit the same local polynomial
with the same kernel to the available observations (Proietti & Luati, 2008, eq. 4). It
reproduces polynomials of the filter's degree, but its real-time weights concentrate on
the last observation and give volatile estimates. Requires `h ≥ degree`; when `h == degree` the fit is
exactly determined and the real-time estimate is the last observation itself.
"""
struct DirectAsymmetric <: AbstractEndpoints end

"""
    MinimumRevision(; degree=1, ratio=:msre)

Asymmetric weights minimising the mean square revision error with respect to the
symmetric filter (Proietti & Luati, 2008, §4), subject to reproducing polynomials of
degree `degree` (`d*`), assuming the signal also has a term `δ jᵈ*⁺¹` and white noise of
variance `σ²`.

- `degree`: `0` (LC: linear signal, constant fit), `1` (QL: quadratic signal, linear
  fit) or `2` (CQ: cubic signal, quadratic fit); must be below both the filter's degree
  and `h`.
- `ratio`: `δ²/σ²` (a non-negative number, or `Inf` for the limit in which degree
  `d* + 1` is also reproduced exactly, which needs `h > d* + 1`), or `:msre` to estimate it in `fit` by minimising
  the mean square revision error of the real-time filter (P&L §5). One ratio is used for
  every end point.

Musgrave's end weights of X-11 are the LC case; see [`musgrave`](@ref).
"""
struct MinimumRevision{R<:Union{Float64,Symbol}} <: AbstractEndpoints
    degree::Int
    ratio::R
end

function MinimumRevision(; degree::Integer=1, ratio::Union{Real,Symbol}=:msre)
    degree ≥ 0 || throw(DomainError(degree, "degree must be non-negative"))
    if ratio isa Symbol
        ratio === :msre ||
            throw(ArgumentError("ratio must be a non-negative number or :msre"))
        return MinimumRevision(Int(degree), ratio)
    end
    ratio ≥ 0 || throw(DomainError(ratio, "ratio must be non-negative"))
    return MinimumRevision(Int(degree), Float64(ratio))
end

"""
    musgrave(R) -> MinimumRevision

Musgrave (1964) end weights used by X-11: [`MinimumRevision`](@ref) with `degree=0` and
`ratio = 4 / (π R²)`, where `R` is the irregular-to-trend (I/C) ratio. X-11 uses
`R = 1.0`, `3.5` and `4.5` with the 9-, 13- and 23-term Henderson filters; see
[`musgrave_ic_ratio`](@ref).
"""
function musgrave(R::Real)
    R > 0 || throw(DomainError(R, "the I/C ratio R must be positive"))
    return MinimumRevision(; degree=0, ratio=4 / (π * R^2))
end

const _MUSGRAVE_IC_RATIOS = Dict(9 => 1.0, 13 => 3.5, 23 => 4.5)

"""
    musgrave_ic_ratio(window) -> Float64

X-11's I/C ratio `R` for the Musgrave end weights of a Henderson filter of length
`window`: `1.0` (9 terms), `3.5` (13 terms) and `4.5` (23 terms). Other lengths default
to `3.5` (not verified against X-13ARIMA-SEATS).
"""

musgrave_ic_ratio(window::Integer) = get(_MUSGRAVE_IC_RATIOS, window, 3.5)

Base.show(io::IO, ::DirectAsymmetric) = print(io, "DirectAsymmetric()")

function Base.show(io::IO, e::MinimumRevision)
    ratio = e.ratio isa Symbol ? repr(e.ratio) : round(e.ratio; sigdigits=4)
    return print(io, "MinimumRevision(degree=$(e.degree), ratio=$ratio)")
end

"""
    LocalPolynomialFilter <: AbstractTrendFilter

Local polynomial trend filter of length `window = 2h + 1`: the weighted least-squares
fit of a polynomial of degree `degree` with kernel weights `kernel` over a centred
window, evaluated at its centre. With the default Henderson kernel and `degree = 3`
this is the Henderson filter of X-11.

Fields: `window`, `degree`, `kernel`, `endpoints` (with any estimated `ratio`),
`weights` (symmetric weights for offsets `j = -h:h`, applied to `yₜ₊ⱼ`) and
`end_weights` (`end_weights[q + 1]` holds the weights for offsets `j = -h:q` at the
point with `q` later observations, `q = 0:h-1`). Create with
`fit(LocalPolynomialFilter, y; ...)`.
"""
struct LocalPolynomialFilter{E<:Union{Nothing,AbstractEndpoints}} <: AbstractTrendFilter
    window::Int
    degree::Int
    kernel::Symbol
    endpoints::E
    weights::Vector{Float64}
    end_weights::Vector{Vector{Float64}}
end

function Base.show(io::IO, f::LocalPolynomialFilter)
    return print(
        io,
        "LocalPolynomialFilter(window=$(f.window), degree=$(f.degree), ",
        "kernel=:$(f.kernel), endpoints=$(repr(f.endpoints)))",
    )
end

"""
    fit(LocalPolynomialFilter, y; window=13, degree=3, kernel=:henderson,
        endpoints=MinimumRevision()) -> LocalPolynomialFilter

Fit a local polynomial trend filter to the series `y` (Proietti & Luati, 2008). The
defaults give the 13-term Henderson filter with quadratic-trend/linear-fit (QL)
minimum-revision end weights whose `ratio` is estimated from `y`.

# Keyword arguments
- `window`: odd filter length `2h + 1`, or `:cv` to choose it by minimising the mean
  leave-one-out cross-validation score `mean(((yₜ - m̂ₜ) / (1 - w₀))²)` over the interior
  points, among the odd lengths in `windows`.
- `windows`: candidate lengths for `window=:cv` (default `2degree + 1` to 51, capped at
  half the series length).
- `degree`: degree of the local polynomial (default `3`).
- `kernel`: kernel weights, see [`filter_kernel`](@ref) (default `:henderson`).
- `endpoints`: [`DirectAsymmetric`](@ref), [`MinimumRevision`](@ref) (e.g.
  [`musgrave`](@ref)) or `nothing` (end points are `NaN`).
"""
function fit(
    ::Type{LocalPolynomialFilter},
    y::AbstractVector{<:Real};
    window::Union{Integer,Symbol}=13,
    windows::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    degree::Integer=3,
    kernel::Symbol=:henderson,
    endpoints::Union{Nothing,AbstractEndpoints}=MinimumRevision(),
)
    y = _check_series(y)
    degree ≥ 0 || throw(DomainError(degree, "degree must be non-negative"))
    if window isa Symbol
        window === :cv || throw(ArgumentError("window must be an odd integer or :cv"))
        windows = something(windows, _default_windows(length(y), degree))
        window = _select_window(y, windows, degree, kernel)
    end
    h = _halfwidth(window, "window")
    _check_length(y, window)
    2h > degree || throw(ArgumentError("window $window is too short for degree $degree"))

    κ = filter_kernel(kernel, h)
    w = _symmetric_weights(κ, degree)
    endpoints = _resolve_endpoints(endpoints, y, κ, w, degree)
    end_weights = _all_end_weights(endpoints, κ, w, degree)
    return LocalPolynomialFilter(
        Int(window), Int(degree), kernel, endpoints, w, end_weights
    )
end

"""
    predict(f::LocalPolynomialFilter, y) -> Vector{Float64}

Apply the filter to `y`: symmetric weights in the interior and `f.end_weights` (mirrored
at the start) in the first and last `h` points.
"""
function predict(f::LocalPolynomialFilter, y::AbstractVector{<:Real})
    y = _check_series(y)
    _check_length(y, f.window)
    N, h = length(y), (f.window - 1) ÷ 2
    x = Vector{Float64}(undef, N)
    for t in (h + 1):(N - h)
        x[t] = dot(f.weights, view(y, (t - h):(t + h)))
    end
    for q in 0:(h - 1)
        v = isnothing(f.endpoints) ? nothing : f.end_weights[q + 1]
        x[N - q] = isnothing(v) ? NaN : dot(v, view(y, (N - q - h):N))
        x[1 + q] = isnothing(v) ? NaN : dot(reverse(v), view(y, 1:(1 + q + h)))
    end
    return x
end

# Polynomial design [(j / s)ᵏ] for k = 0:degree; scaling by s = h keeps it well
# conditioned and leaves the local polynomial fit unchanged
function _design(js::AbstractVector, degree::Integer, s::Real)
    return [(j / s)^k for j in js, k in 0:degree]
end

_unit(n::Integer) = [1.0; zeros(n - 1)]

# Symmetric weights w = KX(XᵀKX)⁻¹e₁ (P&L eq. 3)
function _symmetric_weights(κ::AbstractVector{Float64}, degree::Integer)
    h = (length(κ) - 1) ÷ 2
    X = _design((-h):h, degree, max(h, 1))
    KX = κ .* X
    return KX * ((X' * KX) \ _unit(degree + 1))
end

_all_end_weights(::Nothing, _, _, _) = Vector{Float64}[]

function _all_end_weights(e::AbstractEndpoints, κ, w, degree)
    h = (length(κ) - 1) ÷ 2
    return [_end_weights(e, κ, w, degree, q) for q in 0:(h - 1)]
end

# Direct asymmetric weights w_a = KₚXₚ(XₚᵀKₚXₚ)⁻¹e₁ on offsets -h:q (P&L eq. 4)
function _end_weights(::DirectAsymmetric, κ, _, degree, q)
    h = (length(κ) - 1) ÷ 2
    X = _design((-h):q, degree, h)
    KX = view(κ, 1:(h + 1 + q)) .* X
    return KX * ((X' * KX) \ _unit(degree + 1))
end

# Minimum-revision weights (P&L eqs. 8–9 with D = σ²I and a single bias column Z = jᵈ*⁺¹).
# With a = v - wₚ and ρ = δ²/σ², the first-order conditions are the KKT system
#     [I + ρZₚZₚᵀ  Uₚ] [a]   [ρZₚ(Z_fᵀw_f)]
#     [Uₚᵀ         0 ] [l] = [U_fᵀw_f     ]
# where Uₚ = [jᵏ], k = 0:d*, on the available offsets -h:q (subscript f: the rest)
function _end_weights(e::MinimumRevision{Float64}, κ, w, degree, q)
    isinf(e.ratio) &&    # limit ρ → ∞: degree d* + 1 is also reproduced (P&L eq. 12)
        return _end_weights(MinimumRevision(e.degree + 1, 0.0), κ, w, degree, q)
    h = (length(κ) - 1) ÷ 2
    p, f = 1:(h + 1 + q), (h + 2 + q):(2h + 1)
    js = (-h):h
    U = _design(js, e.degree, h)
    z = @. (js / h)^(e.degree + 1)
    ρ = e.ratio * Float64(h)^(2 * (e.degree + 1))    # ratio for the scaled column z
    Up, zp, wp = U[p, :], z[p], w[p]
    zf_wf = dot(view(z, f), view(w, f))
    k = e.degree + 1
    A = [(I + ρ * zp * zp') Up; Up' zeros(k, k)]
    b = [ρ * zf_wf * zp; U[f, :]' * view(w, f)]
    return wp + (A \ b)[p]
end

function _resolve_endpoints(e::Union{Nothing,AbstractEndpoints}, y, κ, w, degree)
    h = (length(κ) - 1) ÷ 2
    if e isa DirectAsymmetric
        h ≥ degree || throw(
            ArgumentError(
                "direct asymmetric end weights of degree $degree need h ≥ $degree, got h = $h",
            ),
        )
    elseif e isa MinimumRevision
        e.degree < degree || throw(
            ArgumentError(
                "minimum-revision degree ($(e.degree)) must be below the filter degree ($degree)",
            ),
        )
        e.degree < h || throw(
            ArgumentError(
                "minimum-revision degree $(e.degree) needs h > $(e.degree), got h = $h"
            ),
        )
        e.ratio == Inf &&
            !_allows_infinite_ratio(e, h) &&
            throw(
                ArgumentError(
                    "ratio=Inf reproduces degree $(e.degree + 1), which needs h > $(e.degree + 1)",
                ),
            )
        e.ratio === :msre &&
            return MinimumRevision(e.degree, _estimate_ratio(e, y, κ, w, degree))
    end
    return e
end

# The ρ → ∞ limit reproduces degree d* + 1 and must leave a degree of freedom at q = 0
_allows_infinite_ratio(e::MinimumRevision, h::Integer) = e.degree + 1 < h

const _MIN_MSRE_POINTS = 10

# ρ minimising the mean square revision error mean((m̂ₜ - m̂ₜ|ₜ)²) of the real-time
# (q = 0) filter over t = h+1:N-h (P&L §5): grid search over {0, Inf} ∪ 10^(-4:0.1:2),
# refined by Brent's method on log₁₀ρ around the best interior grid point
function _estimate_ratio(e::MinimumRevision, y, κ, w, degree)
    h = (length(κ) - 1) ÷ 2
    N = length(y)
    ts = (h + 1):(N - h)
    length(ts) ≥ _MIN_MSRE_POINTS || throw(
        ArgumentError(
            "estimating ratio=:msre needs a series of length ≥ $(2h + _MIN_MSRE_POINTS); " *
            "pass a numeric ratio instead",
        ),
    )
    final = [dot(w, view(y, (t - h):(t + h))) for t in ts]
    function msre(ρ)
        v = _end_weights(MinimumRevision(e.degree, ρ), κ, w, degree, 0)
        return mean(abs2(final[i] - dot(v, view(y, (t - h):t))) for (i, t) in enumerate(ts))
    end

    log_grid = -4:0.1:2
    grid = [0.0; exp10.(log_grid); _allows_infinite_ratio(e, h) ? [Inf] : Float64[]]
    scores = msre.(grid)
    i = argmin(scores)
    (i == 1 || i > length(log_grid) + 1) && return grid[i]
    lo, hi = log_grid[max(i - 2, 1)], log_grid[min(i, length(log_grid))]
    res = optimize(s -> msre(exp10(s)), lo, hi, Brent())
    return Optim.minimum(res) < scores[i] ? exp10(Optim.minimizer(res)) : grid[i]
end

function _default_windows(N::Integer, degree::Integer)
    upper = min(51, N ÷ 2)
    upper = isodd(upper) ? upper : upper - 1
    return (2degree + 1):2:upper
end

# Window minimising the mean leave-one-out CV score (P&L §5); the mean, not the sum,
# makes scores comparable across windows with different numbers of interior points
function _select_window(y, windows, degree, kernel)
    isempty(windows) && throw(ArgumentError("no candidate windows for window=:cv"))
    N = length(y)
    scores = map(windows) do window
        h = _halfwidth(window, "window")
        N ≥ window && 2h > degree || return Inf
        w = _symmetric_weights(filter_kernel(kernel, h), degree)
        return mean(((h + 1):(N - h))) do t
            return ((y[t] - dot(w, view(y, (t - h):(t + h)))) / (1 - w[h + 1]))^2
        end
    end
    all(isinf, scores) && throw(ArgumentError("no candidate window fits the series"))
    return windows[argmin(scores)]
end
