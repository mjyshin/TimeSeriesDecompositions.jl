struct ComponentEstimate
    mean::Vector{Float64}
    std::Vector{Float64}
    lower::Vector{Vector{Float64}}
    upper::Vector{Vector{Float64}}
    sig::Vector{Float64}
end

struct Decomposition
    t::Vector{Float64}
    trend::ComponentEstimate
    seasonal::ComponentEstimate
    noise::ComponentEstimate
    rate::Union{ComponentEstimate, Nothing}
    params::NamedTuple
    _posterior::Any
    _fit_info::NamedTuple
end

function Base.show(io::IO, d::Decomposition)
    N = length(d.t)
    t1, tN = extrema(d.t)
    sig_str = join(round.(Int, 100 .* d.trend.sig), "%, ") * "%"
    has_rate = !isnothing(d.rate)
    θ = d.params
    println(io, "Decomposition{N=$N, t∈[$t1, $tN]}")
    println(io, "  components : trend, seasonal, noise$(has_rate ? ", rate" : "")")
    println(io, "  CI levels  : $sig_str")
    println(io, "  period     : $(θ.period)")
    println(io, "  bandwidth  : $(isfinite(θ.bandwidth) ? θ.bandwidth : "Inf (static)")")
    println(io, "  σ_trend    : $(round(θ.σ_trend; sigdigits=4))")
    println(io, "  l_trend    : $(round(θ.l_trend; sigdigits=4))")
    println(io, "  σ_seasonal : $(round(θ.σ_seasonal; sigdigits=4))")
    println(io, "  l_seasonal : $(round(θ.l_seasonal; sigdigits=4))")
    print(io,   "  σ_noise    : $(round(θ.σ_noise; sigdigits=4))")
end

function Base.show(io::IO, ::MIME"text/plain", d::Decomposition)
    show(io, d)
end
