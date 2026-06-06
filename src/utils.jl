function demean(y::AbstractVector{<:Real})
    ȳ = mean(y)
    return y .- ȳ, ȳ
end

function standardize(y::AbstractVector{<:Real})
    y_zero, ȳ = demean(y)
    s = std(y_zero)
    return y_zero ./ s, ȳ, s
end

destandardize(y::AbstractVector{<:Real}, ȳ::Real, s::Real) = s .* y .+ ȳ
