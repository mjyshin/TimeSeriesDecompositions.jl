function standardize(y::AbstractVector{<:Real})
    ȳ = mean(y)
    s = std(y .- ȳ)
    return (y .- ȳ) ./ s, ȳ, s
end

destandardize(y::AbstractVector{<:Real}, ȳ::Real, s::Real) = s .* y .+ ȳ
