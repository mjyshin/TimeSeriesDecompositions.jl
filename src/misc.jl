function demean(y::Vector{<:Real})
    ȳ = mean(y)

    return y .- ȳ, ȳ
end

function normalize(y::Vector{<:Real})
    y_zero, ȳ = demean(y)    # zero mean data
    sy = std(y_zero)

    return y_zero/sy, ȳ, sy
end
normalize(θ::Param, sy::Real) = toθ([θ.σ_x̄/sy, θ.l_x̄, θ.σ_x̃/sy, θ.l_x̃, θ.T_x̃, θ.σ_ϵ/sy])
denormalize(θ::Param, sy::Real) = toθ([θ.σ_x̄*sy, θ.l_x̄, θ.σ_x̃*sy, θ.l_x̃, θ.T_x̃, θ.σ_ϵ*sy])

function deslope(y::Vector{<:Real}, t::AbstractVector{<:Real}=1:length(y))
    @assert length(y) == length(t)
    X = [ones(length(t)) t]
    w = (X'*X)\X'*y    # peudoinverse method
    ŷ = X*w

    return y - ŷ, w
end

±(x,y) = (x-y, x+y)
