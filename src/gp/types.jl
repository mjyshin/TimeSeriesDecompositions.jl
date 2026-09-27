struct GPDecomp <: AbstractDecomp
    t::Vector{Float64}
    trend::Component
    seasonal::Union{Component,Nothing}
    noise::Union{Component,Nothing}
    rate::Component
    sig::Vector{Float64}
    θ::NamedTuple
    _posterior::Any
    _fit_info::NamedTuple  # (ȳ, s_y, approx_rate, fit): fit has optimizer diagnostics
end

const _GP_HYPERPARAMETERS = (:σ_trend, :l_trend, :σ_seasonal, :l_seasonal, :σ_noise)

_kernel_name(k) = string(nameof(typeof(k)))
_kernel_name(k::TransformedKernel) = _kernel_name(k.kernel)
_kernel_name(k::ScaledKernel) = _kernel_name(k.kernel)

function Base.show(io::IO, ::MIME"text/plain", d::GPDecomp)
    θ, fi = d.θ, d._fit_info.fit
    has_seasonal = !isnothing(d.seasonal)
    has_noise = !isnothing(d.noise)
    _show_header(io, d)

    comps = join(
        ["trend"; has_seasonal ? ["seasonal"] : []; has_noise ? ["noise"] : []; "rate"],
        ", ",
    )
    _show_line(io, "components", comps)
    ci = join(round.(Int, 100 .* d.sig), "%, ") * "%"
    d._fit_info.approx_rate && (ci *= " (none for rate: approx_rate)")
    _show_line(io, "CI levels", ci)
    _show_line(io, "period", has_seasonal ? _fmt(θ.period) : "none")
    has_seasonal && _show_line(
        io, "bandwidth", isfinite(θ.bandwidth) ? _fmt(θ.bandwidth) : "Inf (static)"
    )

    kernels = Pair{Symbol,String}[:trend => _kernel_name(fi.kernels.trend)]
    if has_seasonal
        k_seasonal = _kernel_name(fi.kernels.seasonal.effect)
        push!(kernels, :seasonal => "$k_seasonal, periodic, zero-mean")
        isfinite(θ.bandwidth) &&
            push!(kernels, :envelope => _kernel_name(fi.kernels.seasonal.envelope))
    end
    _show_section(io, "kernels", kernels)

    # Structurally absent parameters (no seasonal, no noise) are neither estimated nor fixed
    present = filter(_GP_HYPERPARAMETERS) do k
        return (has_seasonal || k ∉ (:σ_seasonal, :l_seasonal)) &&
               (has_noise || k != :σ_noise)
    end
    function row(k)
        v = _fmt(θ[k])
        k == :l_trend && fi.l_trend_min > 0 || return k => v
        return k => "$v (> $(_fmt(fi.l_trend_min))$(fi.at_bound ? ", at bound" : ""))"
    end
    _show_section(io, "estimated", [row(k) for k in present if k ∈ fi.free])
    _show_section(io, "fixed", [row(k) for k in present if k ∉ fi.free])

    fit_str = "log-likelihood $(_fmt(fi.loglik))"
    if !isnothing(fi.optimizer)
        status = fi.converged ? "converged" : "not converged"
        fit_str *= ", $(fi.optimizer) $status in $(fi.iterations) iterations"
    end
    _show_line(io, "fit", fit_str)
    return nothing
end
