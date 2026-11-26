"""
    GaussianProcess(; kernel=Matern52Kernel(), θ_init=(;), θ_fixed=(;), optimizer=LBFGS(...),
                    max_iter=1000)

Gaussian-process decomposition model: trend, seasonal and noise processes whose
hyperparameters maximize the marginal likelihood. Components have posterior means,
standard deviations and credible intervals, in the units of `y`.

- `kernel`: a stationary `Kernel` for every component, or a `NamedTuple` with keys
  `trend` and `seasonal`, the latter a `Kernel` or a `NamedTuple` with keys `effect` and
  `envelope`; unspecified components take `Matern52Kernel()`. Stored resolved and scaled
  to unit variance (the `σ` set the scale), as `(trend, seasonal = (effect, envelope))`.
- `θ_init`: starting values of hyperparameters among `σ_trend`, `l_trend`, `σ_seasonal`,
  `l_seasonal`, `bandwidth`, `σ_noise` (e.g. the `θ` of a previous fit). `σ = 0` and
  `l_trend = Inf` are held fixed.
- `θ_fixed`: hyperparameters held fixed, with the same keys; they override `θ_init`.
- `optimizer`, `max_iter`: Optim.jl optimizer and iteration limit; the gradient of the
  marginal likelihood is exact.

## Identifiability

The seasonal kernel has its mean over a period removed, so the seasonal component cannot
absorb the trend, and `l_trend` is constrained above `period` (a fixed one must exceed it).
`l_seasonal` is capped at 10. A warning is emitted if `l_trend` ends on its bound, or if
the standardized innovations fail a Ljung–Box test at the 1% level
(`d._fit_info.fit.whiteness`).

## Bandwidth estimation

The seasonal envelope has lengthscale `bandwidth ≥ period`, parametrized by
`ρ = period / bandwidth ∈ [0, 1]`; `bandwidth = Inf` is a static pattern. Unless it is
fixed, the static model is fitted together with free fits from `ρ = 0.9` and from the
log-midpoint of that fit's bandwidth and the span of `t` (warm-started from the static
fit). A `θ_init.bandwidth` (e.g. a previous fit's `θ`) replaces these with one free fit
from `θ_init`, or from the static fit if it is `Inf`. The best free fit is kept if
`2(logL - logL_static)` exceeds the 5% critical value 2.71 of its `½χ²₀ + ½χ²₁` null
distribution [1], else `bandwidth = Inf`.
The test is in `d._fit_info.fit.bandwidth_test`, flagged `approximate` below five periods
or when the static fit has `l_seasonal` or `l_trend` on a bound.

[1] S. G. Self and K.-Y. Liang. "Asymptotic properties of maximum likelihood estimators and
likelihood ratio tests under nonstandard conditions". JASA 82 (398), 1987.
"""
struct GaussianProcess{O} <: AbstractDecompositionModel
    kernel::NamedTuple
    θ_init::NamedTuple
    θ_fixed::NamedTuple
    optimizer::O
    max_iter::Int
    function GaussianProcess(kernel, θ_init::NamedTuple, θ_fixed::NamedTuple,
            optimizer::O, max_iter::Integer) where {O}
        θ_init, θ_fixed = _check_θ(θ_init, "θ_init"), _check_θ(θ_fixed, "θ_fixed")
        both = [k for k in keys(θ_init) if haskey(θ_fixed, k)]
        isempty(both) ||
            @warn "θ_init and θ_fixed both give $(join(both, ", ")): using θ_fixed"
        max_iter ≥ 0 || throw(DomainError(max_iter, "max_iter must be non-negative"))
        return new{O}(_resolve_kernels(kernel), θ_init, θ_fixed, optimizer, max_iter)
    end
end

function GaussianProcess(; kernel::Union{Kernel, NamedTuple} = _DEFAULT_KERNEL,
        θ_init::NamedTuple = (;), θ_fixed::NamedTuple = (;),
        optimizer = _default_optimizer, max_iter::Integer = 1_000)
    return GaussianProcess(kernel, θ_init, θ_fixed, optimizer, max_iter)
end

const _GP_HYPERPARAMETERS = (
    :σ_trend, :l_trend, :σ_seasonal, :l_seasonal, :bandwidth, :σ_noise)

function _check_θ(θ::NamedTuple, name)
    _check_keys(θ, _GP_HYPERPARAMETERS, name)
    θ = map(Float64, θ)
    get(θ, :bandwidth, Inf) > 0 ||
        throw(DomainError(θ.bandwidth, "$name.bandwidth must be positive"))
    return θ
end

_kernel_name(k) = string(nameof(typeof(k)))
_kernel_name(k::TransformedKernel) = _kernel_name(k.kernel)
_kernel_name(k::ScaledKernel) = _kernel_name(k.kernel)

function _bandwidth_row(bandwidth, test)
    v = isfinite(bandwidth) ? _fmt(bandwidth) : "Inf (static)"
    isnothing(test) && return v
    v *= " (2ΔlogL vs static = $(round(test.lr_static; digits=1))"
    return v * (test.approximate ? ", approximate test)" : ")")
end

function Base.show(io::IO, ::MIME"text/plain", d::Decomposition{<:GaussianProcess})
    θ, fi, kernel = d.θ, d._fit_info.fit, d.model.kernel
    has_seasonal = !isnothing(d.seasonal)
    has_noise = !isnothing(d.noise)
    _show_header(io, d)

    comps = join(
        ["trend"; has_seasonal ? ["seasonal"] : []; has_noise ? ["noise"] : []; "rate"],
        ", "
    )
    _show_line(io, "components", comps)
    _show_line(io, "period", has_seasonal ? _fmt(d.period) : "none")

    kernels = Pair{Symbol, String}[:trend => _kernel_name(kernel.trend)]
    if has_seasonal
        k_seasonal = _kernel_name(kernel.seasonal.effect)
        push!(kernels, :seasonal => "$k_seasonal, periodic, zero-mean")
        isfinite(θ.bandwidth) &&
            push!(kernels, :envelope => _kernel_name(kernel.seasonal.envelope))
    end
    present = filter(_GP_HYPERPARAMETERS) do k
        return (has_seasonal || k ∉ (:σ_seasonal, :l_seasonal, :bandwidth)) &&
               (has_noise || k != :σ_noise)
    end
    function row(k)
        k == :bandwidth && return k => _bandwidth_row(θ.bandwidth, fi.bandwidth_test)
        v = _fmt(θ[k])
        k == :l_trend && fi.l_trend_min > 0 || return k => v
        return k => "$v (> $(_fmt(fi.l_trend_min))$(fi.at_bound ? ", at bound" : ""))"
    end
    estimated = [row(k) for k in present if k ∈ fi.free]
    fixed = [row(k) for k in present if k ∉ fi.free]
    width = _section_width(kernels, estimated, fixed)
    _show_section(io, "kernels", kernels; width)
    _show_section(io, "estimated", estimated; width)
    _show_section(io, "fixed", fixed; width)

    fit_str = "log-likelihood $(_fmt(fi.loglik))"
    if !isnothing(fi.optimizer)
        status = fi.converged ? "converged" : "not converged"
        fit_str *= ", $(fi.optimizer) $status in $(fi.iterations) iterations"
    end
    _show_line(io, "fit", fit_str)
    return nothing
end
