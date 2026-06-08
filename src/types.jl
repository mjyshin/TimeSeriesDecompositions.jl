struct Component
    mean::Vector{Float64}
    std::Vector{Float64}
    lower::Vector{Vector{Float64}}
    upper::Vector{Vector{Float64}}
    sig::Vector{Float64}
end

struct Decomposition
    t::Vector{Float64}
    trend::Component
    seasonal::Union{Component, Nothing}
    noise::Union{Component, Nothing}
    rate::Component
    sig::Vector{Float64}
    θ::NamedTuple
    _posterior::Any
    _fit_info::NamedTuple  # (ȳ, s_y, approx_rate)
end

function Base.show(io::IO, d::Decomposition)
    N = length(d.t)
    t1, tN = extrema(d.t)
    sig_str = join(round.(Int, 100 .* d.sig), "%, ") * "%"
    has_seasonal = !isnothing(d.seasonal)
    has_noise    = !isnothing(d.noise)
    θ = d.θ
    comps = "trend" * (has_seasonal ? ", seasonal" : "") * (has_noise ? ", noise" : "") * ", rate"
    println(io, "Decomposition{N=$N, t∈[$t1, $tN]}")
    println(io, "  components : $comps")
    println(io, "  CI levels  : $sig_str")
    if has_seasonal
        println(io, "  period     : $(round(θ.period; sigdigits=4))")
        println(io, "  bandwidth  : $(isfinite(θ.bandwidth) ? round(θ.bandwidth; sigdigits=4) : "Inf (static)")")
    end
    println(io, "  σ_trend    : $(round(θ.σ_trend; sigdigits=4))")
    println(io, "  l_trend    : $(round(θ.l_trend; sigdigits=4))")
    if has_seasonal
        println(io, "  σ_seasonal : $(round(θ.σ_seasonal; sigdigits=4))")
        println(io, "  l_seasonal : $(round(θ.l_seasonal; sigdigits=4))")
    end
    if has_noise
        print(io, "  σ_noise    : $(round(θ.σ_noise; sigdigits=4))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", d::Decomposition)
    show(io, d)
end
