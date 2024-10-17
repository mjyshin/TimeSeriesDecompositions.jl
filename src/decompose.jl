function run_static_decomp(
        y::Vector{<:Real}, t::AbstractVector{<:Real};
        t_out::AbstractVector{<:Real}=t, θ::NamedTuple=NamedTuple(), fix::Vector{Symbol}=collect(keys(θ)),
        norm::Bool=norm, logscale::Bool=logscale, β_ci::Union{Real, Vector{<:Real}}=β_ci,
        rate::Bool=rate, fast_rate::Bool=fast_rate, save::Bool=save, δ::Real=δ
    )
    # 1. Optimize parameters and compute posterior
    opt = optim_post(y, t, norm, θ, fix, rate)

    # 2. Compute marginals and series
    ρ_marg = comp_marg(opt.ρ_post, opt.ȳ, opt.s_y, rate=rate, fast_rate=fast_rate, δ=δ)
    decomp = comp_decomp(ρ_marg, t_out, β_ci=β_ci, logscale=logscale)    # decomposed series
    θ = opt.θ    # optimal param
    ρ = save ? (marg=ρ_marg, post=opt.ρ_post) : (marg=ρ_marg,)    # posterior density(/densities)

    return (decomp=decomp, θ=θ, ρ=ρ)
end

function run_dynamic_decomp(
        y::Vector{<:Real}, t::AbstractVector{<:Real}, bw::Real;
        x̃::Union{Nothing, Vector{<:Real}}=nothing, t_out::AbstractVector{<:Real}=t,
        θ::NamedTuple=NamedTuple(), fix::Vector{Symbol}=collect(keys(θ)),
        norm::Bool=norm, logscale::Bool=logscale, β_ci::Union{Real, Vector{<:Real}}=β_ci,
        rate::Bool=rate, fast_rate::Bool=fast_rate, δ::Real=δ
    )
    # Ensure x̄ params are fixed
    if !(:σ_x̄ ∈ fix && :l_x̄ ∈ fix)
        if isnothing(x̃)
            out = run_static_decomp(y, t,
                t_out=t, θ=θ, fix=fix, norm=norm, logscale=logscale,
                β_ci=β_ci, rate=false, fast_rate=false, save=false, δ=δ
            )
            x̃ = out.decomp.x̃.μ
            θ = out.θ
        end
        ȳ = y - x̃    # deseasonalized data
        out = run_static_decomp(ȳ, t,
            t_out=[0], θ=(θ..., σ_x̃=0), fix=unique([fix..., :σ_x̃, :l_x̃, :T_x̃]),
            norm=norm, logscale=logscale, β_ci=β_ci, rate=false, fast_rate=false, save=false, δ=δ
        )
        θ = delete(out.θ, :σ_x̃)
        fix = unique([fix..., :σ_x̄, :l_x̄, :T_x̃])
    end

    # Loop over t_out
    θs = Param[]
    decomp = []
    for tₙ ∈ t_out
        keep = tₙ-bw/2-eps() .≤ t .≤ tₙ+bw/2+eps()
        opt = optim_post(y[keep], t[keep], norm, θ, fix, rate)

        ρ_marg = comp_marg(opt.ρ_post, opt.ȳ, opt.s_y, rate=rate, fast_rate=fast_rate, δ=δ)
        push!(decomp, comp_decomp(ρ_marg, tₙ, β_ci=β_ci, logscale=logscale))
        θ = opt.θ
        push!(θs, θ)
    end

    # Concatenate decompositions
    mkeys = rate || fast_rate ? margkeys′ : margkeys
    decomp = (; (mkeys .=> [getindex.(decomp, key) for key ∈ mkeys])...)
    for (key, value) ∈ pairs(decomp)
        comp = NamedTuple()
        for k ∈ keys(value[1])
            v = getindex.(value, k)
            if (k == :l || k == :u) && length(β_ci) > 1 v = collect.(eachrow(hcat(v...))) end
            comp = (; comp..., k => v)
        end
        decomp = (; decomp..., key => comp)
    end
    decomp = (t=t_out, decomp..., β_ci=β_ci)

    return (decomp=decomp, θ=θs)
end

function decompose(
        y::Vector{<:Real}, t::AbstractVector{<:Real}=1:length(y);
        keep::AbstractVector{Bool}=trues(length(y)), t_out::AbstractVector{<:Real}=copy(t),
        logscale::Bool=logscale, norm::Bool=norm, slope::Bool=slope,    # preprocess
        θ::NamedTuple=NamedTuple(), fix::Vector{<:Union{Symbol, String}}=collect(keys(θ)),    # params
        β_ci::Union{Real, Vector{<:Real}}=β_ci, rate::Bool=rate, fast_rate::Bool=fast_rate, save::Bool=save, δ::Real=δ,
        filter_outliers::Bool=filter_outliers, β_outliers::Real=β_outliers,    # outliers
        dynamic::Bool=dynamic, n_dynamic::Int=n_dynamic, bw::Real=:T_x̃ ∈ keys(θ) ? 6θ.T_x̃ : 0
    )
    # 1. Preprocess data
    @assert length(y) == length(t) == length(keep)
    data = (t=t, y=y)    # original time series
    y = logscale ? log1p.(y[keep]) : y[keep]    # log-transform
    t = t[keep]

    # 2. Preprocess initial parameters
    @assert all(values(θ) .≥ 0)    # ensure positive
    fix = unique(Symbol.(fix))
    for key ∈ θkeys fix = key ∉ keys(θ) ? fix[fix .!= key] : fix end    # don't fix params not specified in θ

    if :σ_x̄ ∈ keys(θ) && iszero(θ.σ_x̄) && :σ_x̄ ∈ fix    # no trend
        θ = (θ..., l_x̄=Inf)
        fix = unique([fix..., :l_x̄])    # no need to optimize other x̄ params
    end
    if :σ_x̃ ∈ keys(θ) && iszero(θ.σ_x̃) && :σ_x̃ ∈ fix    # no seasonality
        θ = (θ..., l_x̃=Inf, T_x̃=Inf)
        fix = unique([fix..., :l_x̃, :T_x̃])    # no need to optimize other x̃ params
    end

    # 3. Run decomposition
    if dynamic
        @assert n_dynamic ≥ 1
        if :σ_x̄ ∈ fix && :l_x̄ ∈ fix n_dynamic = 1 end
        x̃ = nothing
        if n_dynamic > 1
            for _ ∈ 1:n_dynamic-1
                out = run_dynamic_decomp(y, t, bw,
                    x̃=x̃, t_out=t, θ=θ, fix=fix, norm=norm, logscale=logscale,
                    β_ci=β_ci, rate=false, fast_rate=false, δ=δ
                )
                x̃ = out.decomp.x̃.μ
                θ = out.θ[end]
            end
        end
        out = run_dynamic_decomp(y, t, bw,
            x̃=x̃, t_out=t_out, θ=θ, fix=fix, norm=norm, logscale=logscale,
            β_ci=β_ci, rate=rate, fast_rate=fast_rate, δ=δ
        )
    else
        out = run_static_decomp(y, t,
            t_out=t_out, θ=θ, fix=fix, norm=norm, logscale=logscale,
            β_ci=β_ci, rate=rate, fast_rate=fast_rate, save=save, δ=δ
        )
    end

    ϵ = logscale ? out.decomp.ϵ.logμ : out.decomp.ϵ.μ
    α = 1 - β_outliers
    σ_ϵ = dynamic ? getindex.(out.θ, :σ_ϵ) : out.θ.σ_ϵ
    l_ϵ = 0 .- quantile(Normal(), 1-α/2) .* σ_ϵ
    u_ϵ = 0 .+ quantile(Normal(), 1-α/2) .* σ_ϵ
    outliers = .!(l_ϵ .≤ ϵ .≤ u_ϵ)

    return (data=data, out..., outliers=outliers, args=(logscale=logscale, rate=rate, dynamic=dynamic))
    # if !filter_outliers return (comp..., outliers=falses(length(y))) end

    # ϵ = logscale ? log.(comp.series.ϵ.μ) : comp.series.ϵ.μ
    # # ϵ_keep = ϵ[keep]
    # α = 1 - β_outliers
    # # ϵ_filter = mean(ϵ_keep) ± quantile(Normal(), 1-α/2)*std(ϵ_keep)
    # ϵ_filter = 0 ± quantile(Normal(), 1-α/2)*comp.θ.σ_ϵ
    # outliers = .!(ϵ_filter[1] .≤ ϵ .≤ ϵ_filter[2])
    # keep_filter = .!outliers .&& keep

    # comp = run_decomp(y, t, θ=comp.θ, logscale=logscale, norm=norm, fix=fix, keep=keep_filter, β_ci=β_ci, rate=filter_outliers, fast_rate=fast_rate, save=save, δ=δ)

    # return (comp..., outliers=outliers)
end
