function demean(y::Vector{<:Real})
    ȳ = mean(y)

    return y .- ȳ, ȳ
end

function normalize(y::Vector{<:Real})
    y_zero, ȳ = demean(y)    # zero mean data
    s_y = std(y_zero)

    return y_zero/s_y, ȳ, s_y
end
normalize(θ::Param, s_y::Real) = toθ([θ.σ_x̄/s_y, θ.l_x̄, θ.σ_x̃/s_y, θ.l_x̃, θ.T_x̃, θ.σ_ϵ/s_y])
denormalize(y::Vector{<:Real}, ȳ::Real, s_y::Real) = s_y*y .+ ȳ
denormalize(θ::Param, s_y::Real) = toθ([θ.σ_x̄*s_y, θ.l_x̄, θ.σ_x̃*s_y, θ.l_x̃, θ.T_x̃, θ.σ_ϵ*s_y])

function deslope(y::Vector{<:Real}, t::AbstractVector{<:Real}=1:length(y))
    @assert length(y) == length(t)
    b = mean(t)
    perm = sortperm(t)
    m = mean(diff(y[perm]) ./ diff(t[perm]))
    ŷ = m*(t .- b)

    return y - ŷ, b, m
end
reslope(y::Vector{<:Real}, t::AbstractVector{<:Real}, b::Real, m::Real) = y + m*(t .- b)

±(x, y) = (x-y, x+y)

function delete(collection::NamedTuple, key::Vector{Symbol})
    keyminus = setdiff(keys(collection), key)
    colminus = getindex.(Ref(collection), keyminus)

    return (; (keyminus .=> colminus)...)
end
delete(collection::NamedTuple, key::Symbol) = delete(collection, [key])
