function normalize(x::AbstractVector{<:Real})
    x̄ = mean(x)
    x_zero = x .- x̄
    sx = std(x_zero)
    x_norm = x_zero/sx

    return x_norm, x̄, sx
end

±(x,y) = (x-y, x+y)
