function normalize(y::AbstractVector{<:Real})
    ȳ = mean(y)
    y_zero = y .- ȳ    # zero mean data
    sy = std(y_zero)
    y_norm = y_zero/sy    # normalize data

    return y_norm, ȳ, sy
end

function deslope(y::AbstractVector{<:Real}, t::AbstractVector{<:Real}=1:length(y))
    @assert length(y) == length(t)
    X = [ones(length(t)) t]
    w = (X'*X)\X'*y    # peudoinverse method
    ŷ = X*w

    return y - ŷ, w
end

±(x,y) = (x-y, x+y)
