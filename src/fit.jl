const _default_optimizer = LBFGS(;
    alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),
)

function _nll(θ::NamedTuple, t_obs, y_obs::AbstractVector{<:Real})
    gp = _build_gp(θ; rate=false)
    return -logpdf(gp(t_obs, 1e-12), y_obs)
end

function _optimize_params(
    θ₀::NamedTuple, fix::AbstractVector{Symbol},
    t_obs, y_obs::AbstractVector{<:Real};
    optimizer=_default_optimizer, max_iter::Int=1_000,
)
    all(k ∈ fix for k ∈ PARAM_KEYS) && return θ₀

    wrapped = _init_params(θ₀, fix)
    θ₀_flat, unflatten = ParameterHandling.value_flatten(wrapped)
    loss = x -> _nll(unflatten(x), t_obs, y_obs)

    function g!(G, x)
        G .= only(Zygote.gradient(loss, x))
    end

    options = Optim.Options(; iterations=max_iter, show_trace=false)
    result = optimize(loss, g!, θ₀_flat, optimizer, options)
    return unflatten(result.minimizer)
end

function _fit_gp(
    y::AbstractVector{<:Real}, t::AbstractVector{<:Real};
    standardize_data::Bool=true,
    init_params::Union{Nothing, NamedTuple}=nothing,
    fix_params::AbstractVector{Symbol}=Symbol[],
    rate::Bool=false,
    bandwidth::Real=Inf,
    optimizer=_default_optimizer,
    max_iter::Int=1_000,
)
    ȳ = 0.0
    s = 1.0
    y_fit = if standardize_data
        y_s, ȳ, s = standardize(y)
        y_s
    else
        Float64.(y)
    end

    θ₀ = _merge_params(init_params)
    # Always fix bandwidth; it enters as a kernel hyperparameter, not optimized
    bw_fix = isfinite(bandwidth) ? Symbol[] : [:bandwidth]
    θ₀ = (θ₀..., bandwidth=bandwidth)
    fix = unique([fix_params; bw_fix; :bandwidth])

    # Scale amplitude guesses by data std
    θ₀ = _standardize_params(θ₀, s)

    t_obs = GPPPInput(:y, collect(Float64, t))
    θ_opt = _optimize_params(θ₀, fix, t_obs, Float64.(y_fit); optimizer, max_iter)

    # Build posterior on the standardized scale
    gp = _build_gp(θ_opt; rate)
    posterior_gp = posterior(gp(t_obs, 1e-12), Float64.(y_fit))

    # Return params in original scale
    θ_final = _destandardize_params(θ_opt, s)

    return (posterior=posterior_gp, params=θ_final, ȳ=ȳ, s=s)
end
