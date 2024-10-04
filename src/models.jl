θkeys = (:σ_x̄, :l_x̄, :σ_x̃, :l_x̃, :T_x̃, :σ_ϵ)
Param = NamedTuple{θkeys, <:NTuple{6, Real}}
toθ(v::Vector{<:Real}) = (; (θkeys .=> v)...)
toθ(θ::NamedTuple) = toθ([key ∈ keys(θ) ? θ[key] : 1 for key ∈ θkeys])

param_type(θ::Real, isfixed::Bool) = isfixed ? fixed(Float64(θ)) : positive(Float64(θ))

init_params(θ::Param, fix::Vector{<:Symbol}) = (
    σ_x̄ = param_type(θ.σ_x̄, :σ_x̄ ∈ fix),
    l_x̄ = param_type(θ.l_x̄, :l_x̄ ∈ fix),

    σ_x̃ = param_type(θ.σ_x̃, :σ_x̃ ∈ fix),
    l_x̃ = param_type(θ.l_x̃, :l_x̃ ∈ fix),
    T_x̃ = param_type(θ.T_x̃, :T_x̃ ∈ fix),

    σ_ϵ = param_type(θ.σ_ϵ, :σ_ϵ ∈ fix)
)

function build_gp(θ::Param, rate::Bool=rate)
    if rate
        return @gppp let
            # Seasonal-trend decomposition with rate
            x̄ = θ.σ_x̄ * GP(with_lengthscale(SEKernel(), θ.l_x̄))
            x̃ = θ.σ_x̃ * GP(with_lengthscale(SEKernel(), θ.l_x̃) ∘ PeriodicTransform(1/θ.T_x̃))
            ϵ = θ.σ_ϵ * GP(WhiteKernel())
    
            y = x̄ + x̃ + ϵ
            r = derivative(x̄)
        end
    end
    return @gppp let
        # Seasonal-trend decomposition without rate
        x̄ = θ.σ_x̄ * GP(with_lengthscale(SEKernel(), θ.l_x̄))
        x̃ = θ.σ_x̃ * GP(with_lengthscale(SEKernel(), θ.l_x̃) ∘ PeriodicTransform(1/θ.T_x̃))
        ϵ = θ.σ_ϵ * GP(WhiteKernel())

        y = x̄ + x̃ + ϵ
    end
end
