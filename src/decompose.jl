function run_decomp(y::Vector{<:Real}, θ::NamedTuple=NamedTuple();
                    t::AbstractVector{<:Real}=1:length(y), logscale::Bool=true,
                    fix::Vector{<:Union{Symbol, String}}=collect(keys(θ)),
                    keep::AbstractVector{Bool}=trues(length(y)), β_ci::Real=0.95,
                    rate::Bool=false, fast_rate::Bool=false, δ::Real=1e-12)
    @assert length(y) == length(t) == length(keep)
    data = (y=y, t=t)
    y = logscale ? log1p.(y) : y    # okay as long as y >> 0
    fix = unique(Symbol.(fix))

    # Set up inputted series
    𝓉 = GPPPInput(:y, t[keep])
    ȳ = mean(y[keep])
    𝓎 = y[keep] .- ȳ    # zero-mean the series

    # Compute optimal params
    if prod(θkeys .∈ Ref(fix))    # CASE 1: FIX ALL PARAMS
        θ = toθ([θ.σ_x̄, θ.l_x̄, θ.σ_x̃, θ.l_x̃, θ.T_x̃, θ.σ_ϵ])
        @assert prod(values(θ) .≥ 0)
        θ̂ = θ    # no need to optimize
    else    # CASE 2: FIT AT LEAST ONE PARAM
        # Don't fix params that are not specified in θ
        for key ∈ θkeys fix = key ∉ keys(θ) ? fix[fix .!= key] : fix end

        # Initialize params with default := 1
        θ = toθ([key ∈ keys(θ) ? θ[key] : 1 for key ∈ θkeys])
        @assert prod(values(θ) .≥ 0)
        if θ.σ_x̃ == 0 && :σ_x̃ ∈ fix
            push!(fix, :l_x̃, :T_x̃)    # no need to optimize x̃ params
            fix = unique(fix)
        end

        # Optimize params
        θ0 = init_params(θ, Symbol.(fix))
        θ̂ = optimize_loss(θ -> ℒ(θ, 𝓉, 𝓎), θ0)
    end

    # Densities
    ρ_post = posterior(build_gp(θ̂, rate && !fast_rate)(𝓉, 0.0), 𝓎)
    ρ_marg = get_marg(ρ_post, ȳ, rate=rate, fast_rate=fast_rate, δ=δ)
    ρ = (post=ρ_post, marg=ρ_marg)

    series = get_series(ρ_marg, t, β_ci=β_ci, logscale=logscale)

    return (data=data, series=series, θ=θ̂, ρ=ρ)
end

function decompose(y::Vector{<:Real}, θ::NamedTuple=NamedTuple();
                   t::AbstractVector{<:Real}=1:length(y), logscale::Bool=true,
                   fix::Vector{<:Union{Symbol, String}}=collect(keys(θ)),
                   keep::AbstractVector{Bool}=trues(length(y)), β_ci::Real=0.95,
                   filter_outliers::Bool=false, β_outliers::Real=0.95,
                   fast_rate::Bool=false, δ::Real=1e-12)
    comp = run_decomp(y, θ, t=t, logscale=logscale, fix=fix, keep=keep, β_ci=β_ci, rate=!filter_outliers, fast_rate=fast_rate, δ=δ)
    if !filter_outliers return (comp..., outliers=falses(length(y))) end

    ϵ = logscale ? log.(comp.series.ϵ.μ) : comp.series.ϵ.μ
    ϵ_keep = ϵ[keep]
    α = 1 - β_outliers
    ϵ_filter = mean(ϵ_keep) ± quantile(Normal(), 1-α/2)*std(ϵ_keep)
    outliers = .!(ϵ_filter[1] .≤ ϵ .≤ ϵ_filter[2])
    keep_filter = .!outliers .&& keep
    # outliers = findall(outliers)    # index rather than BitVector...?

    comp = run_decomp(y, comp.θ, t=t, logscale=logscale, fix=fix, keep=keep_filter, β_ci=β_ci, rate=filter_outliers, fast_rate=fast_rate, δ=δ)

    return (comp..., outliers=outliers)
end
