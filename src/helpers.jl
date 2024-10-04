margkeys = (:y, :x̄, :x̃, :ϵ)
MarginalDens = NamedTuple{margkeys, <:NTuple{4, Function}}

margkeys′ = (margkeys..., :r)
MarginalDens′ = NamedTuple{margkeys′, <:NTuple{5, Function}}

function optim_post(
        y::Vector{<:Real}, t::AbstractVector{<:Real}, norm::Bool,
        θ::NamedTuple, fix::Vector{Symbol}, rate::Bool
    )
    # 0. Preprocess data and params
    y, ȳ, s_y = norm ? normalize(y) : (demean(y)..., 1)    # normalize the series
    θ = normalize(toθ(θ), s_y)    # normalize params
    θ = toθ(θ)    # init unspecified params with default := 1

    # 1. Set up inputted series
    t = GPPPInput(:y, t)

    # 2. Optimize params
    if !all(θkeys .∈ Ref(fix))    # otherwise no need to optimize
        θ₀ = init_params(θ, fix)
        θ = optimize_loss(θ -> ℒ(θ, t, y), θ₀)
    end

    # 3. Compute posterior density
    ρ_post = posterior(build_gp(θ, rate)(t, 0.0), y)

    return (ȳ=ȳ, s_y=s_y, θ=denormalize(θ, s_y), ρ_post=ρ_post)
end

function comp_marg(
        ρ_post::PosteriorGP, ȳ::Real=ȳ, s_y::Real=s_y;
        rate::Bool=rate, fast_rate::Bool=fast_rate, δ::Real=δ
    )
    ρ_y(t::AbstractVector{<:Real}) = s_y*marginals(ρ_post(GPPPInput(:y, t), δ)) .+ ȳ
    ρ_y(t::Real) = ρ_y([t])[1]
    ρ_x̄(t::AbstractVector{<:Real}) = s_y*marginals(ρ_post(GPPPInput(:x̄, t), δ)) .+ ȳ
    ρ_x̄(t::Real) = ρ_x̄([t])[1]
    ρ_x̃(t::AbstractVector{<:Real}) = s_y*marginals(ρ_post(GPPPInput(:x̃, t), δ))
    ρ_x̃(t::Real) = ρ_x̃([t])[1]
    ρ_ϵ(t::AbstractVector{<:Real}) = s_y*marginals(ρ_post(GPPPInput(:ϵ, t), δ))
    ρ_ϵ(t::Real) = ρ_ϵ([t])[1]

    ρ_marg = (y=ρ_y, x̄=ρ_x̄, x̃=ρ_x̃, ϵ=ρ_ϵ)

    if rate
        ρ_r(t::AbstractVector{<:Real}) = fast_rate ? Normal.(fast_deriv(ρ_x̄, t), 0) :
                                         length(t) == 1 ? [ρ_r(t[1])] :    # see definition below
                                         s_y*marginals(ρ_post(GPPPInput(:r, Float64.(t)), δ))
        ρ_r(t::Real) = fast_rate ? ρ_r([t])[1] : ρ_r([t-eps(), t, t+eps()])[2]    # latter avoids error
        ρ_marg = (ρ_marg..., r=ρ_r)
    end

    return ρ_marg
end

function comp_decomp(
        ρ_marg::MarginalDens, t::Union{Real, AbstractVector{<:Real}};
        β_ci::Union{Real, Vector{<:Real}}=β_ci, logscale::Bool=logscale
    )
    α = 1 .- β_ci
    z = quantile.(Normal(), 1 .- α/2)

    decomp = NamedTuple[]
    for (f, ρ) ∈ zip(keys(ρ_marg), ρ_marg)
        μ = mean.(ρ(t))
        σ = std.(ρ(t))
        bounds = [μ ± ζ*σ for ζ ∈ z]
        l, u = getindex.(bounds, 1), getindex.(bounds, 2)

        if logscale    # undo log1p/log
            logμ, logσ = copy(μ), copy(σ)
            delog = f ∈ [:y, :x̄] ? expm1 : exp
            μ = delog.(μ)
            l, u = [delog.(v) for v ∈ l], [delog.(v) for v ∈ u]
            push!(decomp, (μ=μ, l=l, u=u, logμ=logμ, logσ=logσ))
        else
            push!(decomp, (μ=μ, σ=σ, l=l, u=u))
        end
    end
    decomp = (; t=t, (keys(ρ_marg) .=> decomp)..., β_ci=β_ci)

    return decomp
end
function comp_decomp(ρ_marg::MarginalDens′, t::Union{Real, AbstractVector{<:Real}}; β_ci::Union{Real, Vector{<:Real}}=β_ci, logscale::Bool=logscale)
    decomp = comp_decomp(ρ_marg[margkeys], t, β_ci=β_ci, logscale=logscale)

    α = 1 .- β_ci
    z = quantile.(Normal(), 1 .- α/2)
    μ = mean.(ρ_marg.r(t))
    σ = std.(ρ_marg.r(t))
    bounds = [μ ± ζ*σ for ζ ∈ z]
    l, u = getindex.(bounds, 1), getindex.(bounds, 2)
    decomp = (decomp..., r=(μ=μ, σ=σ, l=l, u=u))

    return decomp
end
