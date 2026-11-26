# Posterior of one additive component: its prior with the data of the observation posterior
# (cov(component, y) = component kernel). AbstractGPs has no public API for this.
function _component_estimate(k::Kernel, data, t, shift, s_y)
    f = PosteriorGP(GP(k), data)
    μ = s_y .* mean(f, t) .+ shift
    return Component(μ, s_y .* sqrt.(max.(var(f, t), 0)))
end

# Posterior of the noise, independent across observations: σ²α and σ² - σ⁴[K⁻¹]ᵢᵢ at the
# fitted observations, the prior of a new observation's noise at other times
function _noise_estimate(σ²::Float64, data, t, s_y)
    (; α, C, x) = data
    if t == x
        μ, v = σ² .* α, σ² .- σ²^2 .* diag(inv(C))
    else
        μ, v = zeros(length(t)), fill(σ², length(t))
    end
    return Component(s_y .* μ, s_y .* sqrt.(max.(v, 0)))
end

# Posterior of the trend's derivative, by central differences of step h in the trend kernel
# (forward-mode AD of Matérn kernels is NaN at coincident points)
function _rate_estimate(k::Kernel, data, t, s_y, h)
    (; α, C, x) = data
    Dk = (kernelmatrix(k, x, t .+ h) .- kernelmatrix(k, x, t .- h)) ./ 2h
    μ = s_y .* (Dk' * α)
    D²k = (kernelmatrix_diag(k, t .+ h) .+ kernelmatrix_diag(k, t .- h) .-
           2 .* kernelmatrix_diag(k, t .+ h, t .- h)) ./ 4h^2
    v = D²k .- vec(sum(abs2, C.L \ Dk; dims = 1))
    return Component(μ, s_y .* sqrt.(max.(v, 0)))
end

function _build_decomposition(model::GaussianProcess, fit_info::NamedTuple,
        t::AbstractVector{<:Real}, period::Float64, θ::NamedTuple)
    t = collect(Float64, t)
    (; posterior, ȳ, s_y) = fit_info
    (; ks, data) = posterior
    estimate(k, shift = 0.0) = isnothing(k) ? nothing :
                               _component_estimate(k, data, t, shift, s_y)
    h = 1e-4 * (isfinite(θ.l_trend) ? θ.l_trend : 1.0)
    rate = _rate_estimate(ks.trend, data, t, s_y, h)
    return Decomposition(model, t, period, estimate(ks.trend, ȳ), estimate(ks.seasonal),
        ks.noise > 0 ? _noise_estimate(ks.noise, data, t, s_y) : nothing, rate, θ,
        nothing, fit_info)
end

"""
    predict(d::Decomposition{<:GaussianProcess}, t_new=d.t)

Evaluate a fitted decomposition at the times `t_new` without refitting. The noise is independent across observations: at `t_new = d.t` it is the
posterior of each observation's noise, elsewhere that of a new observation (mean 0).
"""
function predict(d::Decomposition{<:GaussianProcess}, t_new::AbstractVector{<:Real} = d.t)
    return _build_decomposition(d.model, d._fit_info, t_new, d.period, d.θ)
end
