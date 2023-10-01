θkeys = (:σ_x̄, :l_x̄, :σ_x̃, :l_x̃, :T_x̃, :σ_ϵ)
Param = NamedTuple{θkeys, <:NTuple{6, Real}}
toθ(v::Vector{<:Real}) = (; (θkeys .=> v)...)

function build_gp(θ::Param, rate::Bool=false)
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
