function _component(μ::Vector{Float64}, σ::Vector{Float64}, sig::Vector{Float64})
    z = [quantile(Normal(), (1 + p) / 2) for p in sig]
    return Component(μ, σ, [μ .- ζ .* σ for ζ in z], [μ .+ ζ .* σ for ζ in z], sig)
end

# Posterior of one additive component: its prior with the data of the observation posterior
# (cov(component, y) = component kernel). AbstractGPs has no public API for this.
function _component_estimate(k::Kernel, data, t, sig, shift, s_y)
    f = PosteriorGP(GP(k), data)
    μ = s_y .* mean(f, t) .+ shift
    return _component(μ, s_y .* sqrt.(max.(var(f, t), 0)), sig)
end

# Posterior of the trend's derivative, by central differences of step h in the trend kernel
# (forward-mode AD of Matérn kernels is NaN at coincident points)
function _rate_estimate(k::Kernel, data, t, sig, s_y, h; approx::Bool)
    (; α, C, x) = data
    Dk = (kernelmatrix(k, x, t .+ h) .- kernelmatrix(k, x, t .- h)) ./ 2h
    μ = s_y .* (Dk' * α)
    approx && return _component(μ, fill(NaN, length(t)), sig)
    D²k = (kernelmatrix_diag(k, t .+ h) .+ kernelmatrix_diag(k, t .- h) .-
           2 .* kernelmatrix_diag(k, t .+ h, t .- h)) ./ 4h^2
    v = D²k .- vec(sum(abs2, C.L \ Dk; dims = 1))
    return _component(μ, s_y .* sqrt.(max.(v, 0)), sig)
end

function _build_decomposition(post::NamedTuple, t_out::Vector{Float64}, θ::NamedTuple,
        ȳ::Float64, s_y::Float64; sig::Vector{Float64}, approx_rate::Bool, fit_info)
    sig = sort(unique(sig))
    (; ks, data) = post
    estimate(k, shift = 0.0) = isnothing(k) ? nothing :
                               _component_estimate(k, data, t_out, sig, shift, s_y)
    h = 1e-4 * (isfinite(θ.l_trend) ? θ.l_trend : 1.0)
    rate = _rate_estimate(ks.trend, data, t_out, sig, s_y, h; approx = approx_rate)
    fi = (; ȳ, s_y, approx_rate, fit = fit_info)
    return GPDecomp(t_out, estimate(ks.trend, ȳ), estimate(ks.seasonal),
        estimate(ks.noise), rate, sig, θ, post, fi)
end

"""
    predict(d::GPDecomp, t_new=d.t; sig=d.sig) -> GPDecomp

Evaluate a fitted decomposition at the times `t_new` without refitting.
"""
function predict(d::GPDecomp, t_new::AbstractVector{<:Real} = d.t;
        sig::AbstractVector{<:Real} = d.sig)
    fi = d._fit_info
    return _build_decomposition(d._posterior, Float64.(t_new), d.θ, fi.ȳ, fi.s_y;
        sig = Float64.(sig), approx_rate = fi.approx_rate, fit_info = fi.fit)
end
