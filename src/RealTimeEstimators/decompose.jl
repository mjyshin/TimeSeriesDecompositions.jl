###################################
###                               #
##    DECOMPOSITION FUNCTIONS    ##
#                               ###
###################################

#== Seasonal decomposition ==#

"""
    deseasonalize(y::Vector{<:Real}, T::Int; Fs::Function=snm, Ft::Function=mrf, h::Int=2T, kwargs...)

Perform a seasonal decomposition of the time series (yₜ) as sum of the seasonal model (x̃ₜ) and deseasonalized data (ȳₜ).
"""
function deseasonalize(y::Vector{<:Real}, T::Int;
        Ft::Function=sma, h::Int=Ft == sma ? T : 2T,
        x̄::Union{Nothing, Vector{<:Real}}=nothing,
        Fs::Function=ssf, logscale::Bool=false,
        kwargs...
    )
    if logscale y = log1p.(y) end

    # Initial trend model
    x̄ = isnothing(x̄) ? Ft(y, h; kwargs...) : logscale ? log.(x̄) : x̄

    # Detrend series / Get seasonal data
    ỹ = y - x̄    # ỹₜ = x̃ₜ + ϵₜ (i.e., seasonal data = seasonal model + noise)

    # Seasonal model
    if :ignore ∈ keys(kwargs)
        ignore = kwargs[:ignore]
        ỹ′ = copy(ỹ)    # non-masked (original) data
        ỹ[ignore] .= NaN    # masked (corrected) data

        x̃′ = Fs(ỹ′, T; kwargs...)    # non-masked (original) model
        x̃ = Fs(ỹ, T; kwargs...)    # masked (corrected) model
        x̃[ignore] .= x̃′[ignore]    # keep non-masked holidays
    else
        x̃ = Fs(ỹ, T; kwargs...)
    end

    # Deseasonalize series / Get trend data
    ȳ = y - x̃    # ȳₜ = x̄ₜ + ϵₜ (i.e., trend data = 𝑡𝑟𝑢𝑒 trend model + noise)

    seas = logscale ? (ȳ=expm1.(ȳ), x̃=exp.(x̃)) : (ȳ=ȳ, x̃=x̃)

    return seas
end


#== Trend decomposition ==#

"""
    decompose(y::Vector{<:Real}, h::Int; Ft::Function=mrf, seasonal::Bool=false, kwargs...)

Perform a (seasonal-)trend decomposition of the time series (yₜ) as sum of the (seasonal model (x̃ₜ),) trend model (x̄ₜ) and noise (ϵₜ).
"""
function decompose(y::Vector{<:Real}, h::Int;
        Ft::Function=mrf, logscale::Bool=false, seasonal::Bool=false,
        kwargs...
    )
    if logscale y = log1p.(y) end
    if !seasonal && :T ∈ keys(kwargs)
        seasonal = true
    end

    # Deseasonalize series
    if seasonal
        T = kwargs[:T]    # period required
        Ft′ = :Ft′ ∈ keys(kwargs) ? kwargs[:Ft′] : sma    # default initial trend filter: MA
        h′ = :h′ ∈ keys(kwargs) ? kwargs[:h′] : Ft′ == sma ? T : 2T
        Fs = :Fs ∈ keys(kwargs) ? kwargs[:Fs] : ssf    # default seasonal filter: SSF
        x̄′ = :x̄′ ∈ keys(kwargs) ? logscale ? log1p.(kwargs[:x̄′]) : kwargs[:x̄′] : nothing

        seas = deseasonalize(y, T, Ft=Ft′, h=h′, Fs=Fs, x̄=x̄′; kwargs...)    # run deseasonalization
        ȳ = seas.ȳ
        x̃ = logscale ? exp.(seas.x̃) : seas.x̃
    else
        ȳ = y
    end

    # Detrend series
    x̄ = Ft(ȳ, h; kwargs...)
    ϵ = ȳ - x̄

    if logscale
        x̄ = exp.(x̄)
        ϵ = exp.(ϵ)
    end

    # Return named tuple
    comp = seasonal ? (x̄=x̄, x̃=x̃, ϵ=ϵ) : (x̄=x̄, ϵ=ϵ)

    return comp
end


#== Seasonal decomposition ==#

"""
    x11(y::Vector{<:Real}, h::Int, T::Int; kwargs...)

Perform a X-11 seasonal-trend decomposition on the time series (yₜ) as sum of the (seasonal model (x̃ₜ),) trend model (x̄ₜ) and noise (ϵₜ). (See https://www.mathworks.com/help/econ/seasonal-adjustment-1.html for details.)
"""
function x11(y::Vector{<:Real}, h::Int, T::Int;
        logscale::Bool=false, Ft::Function=mrf, n::Int=3, m::Int=5,
        h′::Int=h, Ft′::Function=daf, n′::Int=3, m′::Int=3
    )
    if logscale y = log1p.(y) end

    # Decomposition 1 (MA + S₃ₓ₃ + Ft′ [DAF])
    x̄′ = decompose(y, h′,
        Ft′=sma,    # MA filter with bw ≥ T
        T=T, Fs=snm, n=n′, m=m′,    # S₃ₓ₃ seasonal filter with period T
        Ft=Ft′    # asymmetric Henderson filter w/ bw h′
    ).x̄

    # Decomposition 2 (S₃ₓ₅ + Ft [MRF])
    comp = decompose(y, h,
        x̄′=x̄′,    # trend from init decomp
        T=T, Fs=snm, n=n, m=m,    # S₃ₓ₅ seasonal filter with period T
        Ft=Ft    # MR filter with w/ bw h
    )

    # Return named tuple
    if logscale comp = (x̄=expm1.(comp.x̄), x̃=exp.(comp.x̃), ϵ=exp.(comp.ϵ)) end

    return comp
end
