param_type(θ::Real, isfixed::Bool) = isfixed ? fixed(Float64(θ)) : positive(Float64(θ))

init_params(θ::Param, fix::Vector{<:Symbol}) = (
    σ_x̄ = param_type(θ.σ_x̄, :σ_x̄ ∈ fix),
    l_x̄ = param_type(θ.l_x̄, :l_x̄ ∈ fix),

    σ_x̃ = param_type(θ.σ_x̃, :σ_x̃ ∈ fix),
    l_x̃ = param_type(θ.l_x̃, :l_x̃ ∈ fix),
    T_x̃ = param_type(θ.T_x̃, :T_x̃ ∈ fix),

    σ_ϵ = param_type(θ.σ_ϵ, :σ_ϵ ∈ fix)
)

function get_marg(ρ_post::PosteriorGP, ȳ::Real=0; rate::Bool=true, fast_rate::Bool=false, δ::Real=1e-12)
    ρ_y(t::AbstractVector{<:Real}) = marginals(ρ_post(GPPPInput(:y, t), δ)) .+ ȳ
    ρ_y(t::Real) = ρ_y([t])[1]
    ρ_x̄(t::AbstractVector{<:Real}) = marginals(ρ_post(GPPPInput(:x̄, t), δ)) .+ ȳ
    ρ_x̄(t::Real) = ρ_x̄([t])[1]
    ρ_x̃(t::AbstractVector{<:Real}) = marginals(ρ_post(GPPPInput(:x̃, t), δ))
    ρ_x̃(t::Real) = ρ_x̃([t])[1]
    ρ_ϵ(t::AbstractVector{<:Real}) = marginals(ρ_post(GPPPInput(:ϵ, t), δ))
    ρ_ϵ(t::Real) = ρ_ϵ([t])[1]

    ρ_marg = (y=ρ_y, x̄=ρ_x̄, x̃=ρ_x̃, ϵ=ρ_ϵ)

    if rate
        ρ_r(t::AbstractVector{<:Real}) = fast_rate ? Normal.(fast_deriv(ρ_x̄, t), 0) : marginals(ρ_post(GPPPInput(:r, Float64.(t)), δ))
        # ρ_r(t::Real) = ρ_r([t])[1]    # errors out... for fast_rate == false
        ρ_r(t::Real) = ρ_r([t-eps(), t, t+eps()])[2]
        ρ_marg = (ρ_marg..., r=ρ_r)
    end

    return ρ_marg
end

margkeys = (:y, :x̄, :x̃, :ϵ)
MarginalDens = NamedTuple{margkeys, <:NTuple{4, Function}}

margkeys′ = (margkeys..., :r)
MarginalDens′ = NamedTuple{margkeys′, <:NTuple{5, Function}}

±(x,y) = (x-y, x+y)

function get_series(ρ_marg::MarginalDens, t::AbstractVector{<:Real}; β_ci::Real=0.95, logscale::Bool=true)
    y, x̄, x̃, ϵ = [mean.(ρ(t)) for ρ ∈ ρ_marg]
    σ_y, σ_x̄, σ_x̃, σ_ϵ = [std.(ρ(t)) for ρ ∈ ρ_marg]

    α = 1 - β_ci
    z = quantile(Normal(), 1-α/2)
    l_y, u_y = y ± z*σ_y
    l_x̄, u_x̄ = x̄ ± z*σ_x̄
    l_x̃, u_x̃ = x̃ ± z*σ_x̃
    l_ϵ, u_ϵ = ϵ ± z*σ_ϵ

    if logscale
        y, l_y, u_y = [expm1.(v) for v ∈ (y, l_y, u_y)]
        x̄, l_x̄, u_x̄ = [expm1.(v) for v ∈ (x̄, l_x̄, u_x̄)]
        x̃, l_x̃, u_x̃ = [exp.(v) for v ∈ (x̄, l_x̄, u_x̄)]
        ϵ, l_ϵ, u_ϵ = [exp.(v) for v ∈ (ϵ, l_ϵ, u_ϵ)]
    end

    series = (y=(μ=y, l=l_y, u=u_y),
              x̄=(μ=x̄, l=l_x̄, u=u_x̄),
              x̃=(μ=x̃, l=l_x̃, u=u_x̃),
              ϵ=(μ=ϵ, l=l_ϵ, u=u_ϵ))

    return series
end
function get_series(ρ_marg::MarginalDens′, t::AbstractVector{<:Real}; β_ci::Real=0.95, logscale::Bool=true)
    series = get_series(ρ_marg[margkeys], t, β_ci=β_ci, logscale=logscale)

    α = 1 - β_ci
    z = quantile(Normal(), 1-α/2)
    r = mean.(ρ_marg.r(t))
    σ_r = std.(ρ_marg.r(t))
    l_r, u_r = r ± z*σ_r
    series = (series..., r=(μ=r, l=l_r, u=u_r))

    return series
end
