margkeys = (:y, :x̄, :x̃, :ϵ)
MarginalDens = NamedTuple{margkeys, <:NTuple{4, Function}}

margkeys′ = (margkeys..., :r)
MarginalDens′ = NamedTuple{margkeys′, <:NTuple{5, Function}}

function get_marg(ρ_post::PosteriorGP, ȳ::Real=ȳ, sy::Real=sy; rate::Bool=rate, fast_rate::Bool=fast_rate, δ::Real=δ)
    ρ_y(t::AbstractVector{<:Real}) = sy*marginals(ρ_post(GPPPInput(:y, t), δ)) .+ ȳ
    ρ_y(t::Real) = ρ_y([t])[1]
    ρ_x̄(t::AbstractVector{<:Real}) = sy*marginals(ρ_post(GPPPInput(:x̄, t), δ)) .+ ȳ
    ρ_x̄(t::Real) = ρ_x̄([t])[1]
    ρ_x̃(t::AbstractVector{<:Real}) = sy*marginals(ρ_post(GPPPInput(:x̃, t), δ))
    ρ_x̃(t::Real) = ρ_x̃([t])[1]
    ρ_ϵ(t::AbstractVector{<:Real}) = sy*marginals(ρ_post(GPPPInput(:ϵ, t), δ))
    ρ_ϵ(t::Real) = ρ_ϵ([t])[1]

    ρ_marg = (y=ρ_y, x̄=ρ_x̄, x̃=ρ_x̃, ϵ=ρ_ϵ)

    if rate
        ρ_r(t::AbstractVector{<:Real}) = fast_rate ? Normal.(fast_deriv(ρ_x̄, t), 0) :
                                         length(t) == 1 ? [ρ_r(t[1])] :    # see definition below
                                         sy*marginals(ρ_post(GPPPInput(:r, Float64.(t)), δ))
        ρ_r(t::Real) = fast_rate ? ρ_r([t])[1] : ρ_r([t-eps(), t, t+eps()])[2]    # latter avoids error
        ρ_marg = (ρ_marg..., r=ρ_r)
    end

    return ρ_marg
end

function get_series(ρ_marg::MarginalDens, t::Union{Real, AbstractVector{<:Real}}; β_ci::Real=β_ci, logscale::Bool=logscale)
    α = 1 - β_ci
    z = quantile(Normal(), 1-α/2)

    series = NamedTuple[]
    for (f, ρ) ∈ zip(keys(ρ_marg), ρ_marg)
        μ = mean.(ρ(t))
        σ = std.(ρ(t))
        l, u = μ ± z*σ

        if logscale    # undo log1p/log
            if f ∈ [:y, :x̄] μ, l, u = [expm1.(v) for v ∈ [μ, l, u]]
            elseif f ∈ [:x̃, :ϵ] μ, l, u = [exp.(v) for v ∈ [μ, l, u]] end
        end
        push!(series, (μ=μ, l=l, u=u))
    end
    series = (; t=t, (keys(ρ_marg) .=> series)...)

    return series
end
function get_series(ρ_marg::MarginalDens′, t::Union{Real, AbstractVector{<:Real}}; β_ci::Real=β_ci, logscale::Bool=logscale)
    series = get_series(ρ_marg[margkeys], t, β_ci=β_ci, logscale=logscale)

    α = 1 - β_ci
    z = quantile(Normal(), 1-α/2)
    μ = mean.(ρ_marg.r(t))
    σ = std.(ρ_marg.r(t))
    l, u = μ ± z*σ
    series = (series..., r=(μ=μ, l=l, u=u))

    return series
end
