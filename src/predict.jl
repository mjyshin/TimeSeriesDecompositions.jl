function _component_estimate(
    post::PosteriorGP,
    key::Symbol,
    t::AbstractVector{Float64},
    sig::AbstractVector{Float64},
    ȳ::Float64,
    s_y::Float64,
    add_mean::Bool,
)
    jitter = 1e-12
    fg = post(GPPPInput(key, t), jitter)
    μ_std = mean(fg)
    σ_std = sqrt.(max.(0.0, var(fg)))

    μ = s_y .* μ_std .+ (add_mean ? ȳ : 0.0)
    σ = s_y .* σ_std

    z = map(sig) do p
        quantile(Normal(), (1 + p) / 2)
    end
    lower = [μ .- ζ .* σ for ζ ∈ z]
    upper = [μ .+ ζ .* σ for ζ ∈ z]

    return Component(μ, σ, lower, upper, sig)
end

function _approx_rate(
    post::PosteriorGP,
    t::AbstractVector{Float64},
    sig::AbstractVector{Float64},
    s_y::Float64,
)
    span_t = length(t) > 1 ? maximum(t) - minimum(t) : 1.0
    h = max(span_t, 1.0) * 1e-6
    μ_r = map(t) do ti
        μ_hi = mean(post(GPPPInput(:x̄, [ti + h]), 1e-12))[1]
        μ_lo = mean(post(GPPPInput(:x̄, [ti - h]), 1e-12))[1]
        s_y * (μ_hi - μ_lo) / (2h)
    end
    nan_vec = fill(NaN, length(t))
    return Component(μ_r, nan_vec, [copy(nan_vec) for _ in sig], [copy(nan_vec) for _ in sig], sig)
end

# Derivative posterior via posterior covariance finite differences.
# Var[f'(t)] ≈ (K_post(t-h,t-h) + K_post(t+h,t+h) - 2K_post(t-h,t+h)) / (4h²)
function _full_rate(
    post::PosteriorGP,
    t::AbstractVector{Float64},
    sig::AbstractVector{Float64},
    s_y::Float64,
)
    span_t = length(t) > 1 ? maximum(t) - minimum(t) : 1.0
    h = max(span_t, 1.0) * 1e-5
    n = length(t)

    t_paired = [t .- h; t .+ h]   # [t_lo; t_hi], length 2n
    fg = post(GPPPInput(:x̄, t_paired), 1e-12)
    μ_all = mean(fg)
    K = cov(fg)   # 2n × 2n

    μ_r = s_y .* (μ_all[n+1:end] .- μ_all[1:n]) ./ (2h)

    var_lo    = diag(K[1:n,     1:n    ])
    var_hi    = diag(K[n+1:end, n+1:end])
    cov_cross = diag(K[1:n,     n+1:end])
    σ_r = s_y .* sqrt.(max.(0.0, var_lo .+ var_hi .- 2 .* cov_cross) ./ (4h^2))

    z = [quantile(Normal(), (1 + p) / 2) for p in sig]
    lower = [μ_r .- ζ .* σ_r for ζ in z]
    upper = [μ_r .+ ζ .* σ_r for ζ in z]
    return Component(μ_r, σ_r, lower, upper, sig)
end

function _build_decomposition(
    post::PosteriorGP,
    t_out::AbstractVector{Float64},
    params::NamedTuple,
    ȳ::Float64,
    s_y::Float64;
    sig::AbstractVector{Float64},
    approx_rate::Bool,
)
    sig = sort(unique(sig))
    has_seasonal = isfinite(params.period)

    trend    = _component_estimate(post, :x̄, t_out, sig, ȳ, s_y, true)
    seasonal = has_seasonal ?
        _component_estimate(post, :x̃, t_out, sig, 0.0, s_y, false) : nothing
    has_noise = params.σ_noise > 0
    noise    = has_noise ? _component_estimate(post, :ϵ, t_out, sig, 0.0, s_y, false) : nothing
    _rate_estimate = approx_rate ? _approx_rate : _full_rate
    rate     = _rate_estimate(post, t_out, sig, s_y)

    fi = (ȳ=ȳ, s_y=s_y, approx_rate=approx_rate)
    return Decomposition(t_out, trend, seasonal, noise, rate, sig, params, post, fi)
end

"""
    predict(d::Decomposition, t_new=d.t; sig=d.sig) -> Decomposition

Re-evaluate a fitted decomposition at new time points without re-fitting.
"""
function predict(
    d::Decomposition,
    t_new::AbstractVector{<:Real}=d.t;
    sig::AbstractVector{<:Real}=d.sig,
)
    fi = d._fit_info
    return _build_decomposition(
        d._posterior, Float64.(t_new), d.θ,
        fi.ȳ, fi.s_y;
        sig=Float64.(sig), approx_rate=fi.approx_rate,
    )
end
