const _FILTER_KERNELS = (:henderson, :biweight, :tricube, :epanechnikov, :uniform)

"""
    filter_kernel(name::Symbol, h::Integer) -> Vector{Float64}

Unnormalised kernel weights `κⱼ`, `j = -h:h`, for a local polynomial filter of length
`2h + 1`. Valid names are `:henderson`, `:biweight`, `:tricube`, `:epanechnikov` and
`:uniform`.

The Henderson kernel is `κⱼ = [(h+1)² - j²][(h+2)² - j²][(h+3)² - j²]` (Henderson,
1916); with a local cubic it gives the weights that minimise the variance of the third
differences of the filtered series. The other kernels are evaluated at `j / (h + 1)`, so
that the outermost weights are non-zero.
"""
function filter_kernel(name::Symbol, h::Integer)
    h ≥ 0 || throw(DomainError(h, "h must be non-negative"))
    j = (-h):h
    u = j ./ (h + 1)
    κ = if name === :henderson
        @. ((h + 1)^2 - j^2) * ((h + 2)^2 - j^2) * ((h + 3)^2 - j^2)
    elseif name === :biweight
        @. (1 - u^2)^2
    elseif name === :tricube
        @. (1 - abs(u)^3)^3
    elseif name === :epanechnikov
        @. 1 - u^2
    elseif name === :uniform
        ones(length(j))
    else
        throw(ArgumentError("unknown kernel :$name; valid kernels: $_FILTER_KERNELS"))
    end
    return Float64.(κ)
end
