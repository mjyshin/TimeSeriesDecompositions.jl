#########################################
###                                     #
##    SEASONAL FILTERING ALGORITHMS    ##
#                                     ###
#########################################

#== Stable seasonal filter (SSF) ==#

τ(t::Int, T::Int) = mod(t-1, T) + 1    # clock function

"""
    ssf(ỹ::Vector{<:Real}, T::Int; kwargs...)

Apply the stable seasonal filter (SSF) to the detrended time series (ỹₜ).
"""
function ssf(ỹ::Vector{<:Real}, T::Int; kwargs...)
    @assert T ≥ 1
    # No filtering case
    if T == 1 return ỹ end

    N = length(ỹ)
    ȳ = [mean(filter(!isnan, ỹ[τ:T:N])) for τ ∈ 1:T]    # average over every τ(t) ∈ {1, ..., T}
    x̃ = ȳ[τ.(1:N, T)]    # extend to t = 1, ..., N

    return x̃
end


#== S(n,m) seasonal filter ==#

"""
    snm(ỹ::Vector{<:Real}, T::Int; n::Int=3, m::Int=5, kwargs...)

Apply Sₙₓₘ, a.k.a. S(n, m), seasonal filter to the detrended time series (ỹₜ).
"""
function snm(ỹ::Vector{<:Real}, T::Int; n::Int=3, m::Int=5, kwargs...)
    @assert T ≥ 1
    # No filtering case
    if T == 1 return ỹ end

    N = length(ỹ)
    x̃ = fill(NaN, N)
    for τ ∈ 1:T
        ỹ_τ = ỹ[τ:T:N]    # seasonal data at t s.t. τ(t) = τ

        ȳ_τ = sma(ỹ_τ, m)    # inner moving average
        x̃[τ:T:N] = sma(ȳ_τ, n)    # outer moving average
    end

    return x̃
end
