"""
    MovingAverage(; bandwidth=nothing, endpoints=NoEndpoints(), gapfactor=3)
        <: AbstractTrendSmoother

Gasser–Müller box average over windows of width `bandwidth` (in units of `t`): observation
`i` has weight `|cellᵢ ∩ window| / bandwidth`, with `cellᵢ` its Voronoi cell (tied times
share one). On a grid of spacing `Δ` with `bandwidth = mΔ` it is the `m`-term (odd `m`) or
centered `2 × m` (even `m`) moving average, which annihilates a fixed seasonal pattern of
period `m`.

- `bandwidth`: window width in units of `t`; inside a decomposition, `nothing` gives the
  period.
- `endpoints`: [`NoEndpoints`](@ref) gives `NaN` where the window leaves the outer cells; on
  a grid, [`MinimumRevision`](@ref)`(degree=0)` or [`DirectAsymmetric`](@ref) fill them.
- `gapfactor`: the average is `NaN` where one cell covers more than `gapfactor` typical
  spacings of the window (the length-weighted median gap).

!!! warning
    Values set to `NaN` because of gaps are reported with a single warning (log group
    `:diagnostics`), not an error.
"""
struct MovingAverage{E <: AbstractEndpoints} <: AbstractTrendSmoother
    bandwidth::Union{Nothing, Float64}
    endpoints::E
    gapfactor::Float64
end

function MovingAverage(;
        bandwidth::Union{Nothing, Real} = nothing,
        endpoints::AbstractEndpoints = NoEndpoints(),
        gapfactor::Real = 3
)
    isnothing(bandwidth) ||
        bandwidth > 0 ||
        throw(DomainError(bandwidth, "bandwidth must be positive"))
    gapfactor ≥ 1 || throw(DomainError(gapfactor, "gapfactor must be at least 1"))
    return MovingAverage(_float(bandwidth), endpoints, Float64(gapfactor))
end

function Base.show(io::IO, m::MovingAverage)
    print(io, "MovingAverage(bandwidth=$(_fmt_bandwidth(m.bandwidth)), ")
    print(io, "endpoints=$(m.endpoints)")
    m.gapfactor == 3 || print(io, ", gapfactor=", _fmt(m.gapfactor))
    return print(io, ")")
end

function _fit(model::MovingAverage, y::AbstractVector{<:Real}, t::AbstractVector{<:Real})
    y, t = _sorted_data(y, t)
    isnothing(model.bandwidth) &&
        throw(ArgumentError("MovingAverage needs a bandwidth (in units of t)"))
    W = model.bandwidth
    times, means = _pool_ties(t, y)
    length(times) ≥ 2 || throw(ArgumentError("need at least two distinct times"))
    spacing = _typical_spacing(times)
    estimated = (;)
    endpoints = model.endpoints
    grid = nothing
    if !(endpoints isa NoEndpoints)
        Δ = _spacing(t)
        isnothing(Δ) &&
            throw(ArgumentError("end weights of a MovingAverage need uniformly spaced t"))
        grid = _box_kernel(W / Δ)
        endpoints, estimated = _resolve_endpoints(endpoints, grid, y, 0)
    end
    state = (;
        times,
        means,
        edges = _cell_edges(times),
        limit = model.gapfactor * spacing / W,
        kernel = grid,
        endpoints,
        degree = 0,
        Δ = isnothing(grid) ? nothing : _spacing(t)
    )
    resolved = MovingAverage(W, endpoints, model.gapfactor)
    values, gaps = _box_average(state, W, y, t)
    _warn_gaps(gaps)
    return FittedSmoother(resolved, t, y, values, state, estimated)
end

function predict(f::FittedSmoother{<:MovingAverage}, t_new::AbstractVector{<:Real})
    _check_in_range(f.t, t_new)
    values, gaps = _box_average(f.state, f.model.bandwidth, f.y, t_new, f.values)
    _warn_gaps(gaps)
    return values
end

function _pool_ties(t, y)
    times = unique(t)
    means = [mean(y[searchsortedfirst(t, u):searchsortedlast(t, u)]) for u in times]
    return times, means
end

# Length-weighted median gap: the gap containing a random time
function _typical_spacing(times)
    gaps = sort(diff(times))
    cumulative = cumsum(gaps)
    return gaps[searchsortedfirst(cumulative, last(cumulative) / 2)]
end

# Voronoi cell boundaries, the outer cells extending half a spacing beyond the data
function _cell_edges(times)
    mids = (times[1:(end - 1)] .+ times[2:end]) ./ 2
    return [times[1] - (times[2] - times[1]) / 2;
            mids;
            times[end] + (times[end] - times[end - 1]) / 2]
end

# Box average on a grid as the local constant fit with weights |[d - ½, d + ½] ∩ [-b, b]|
function _box_kernel(m::Real)
    b = m / 2
    return _GridKernel(d -> clamp(b + 0.5 - abs(d), 0.0, 1.0), ceil(Int, b + 0.5) - 1)
end

# Box averages at the times xs and the number of NaNs caused by gaps
function _box_average(state, W, y, xs, grid_values = nothing)
    (; times, means, edges, limit) = state
    gaps = 0
    values = Vector{Float64}(undef, length(xs))
    for (j, x) in enumerate(xs)
        if _interpolate_end(state, W, x, grid_values)
            values[j] = _lagrange(grid_values, (x - first(times)) / state.Δ + 1)
            continue
        end
        a, b = x - W / 2, x + W / 2
        if a < first(edges) - 1e-9 * W || b > last(edges) + 1e-9 * W
            if isnothing(state.kernel)
                values[j] = NaN
            else
                c = (x - first(times)) / state.Δ + 1
                isapprox(c, round(c); atol = 1e-9) && (c = round(c))
                values[j] = _evaluate(
                    state.kernel, state.endpoints, 0, y, grid_values, c, length(y)
                )
            end
            continue
        end
        lo = max(searchsortedlast(edges, a), 1)
        hi = min(searchsortedfirst(edges, b) - 1, length(times))
        total, largest = 0.0, 0.0
        for i in lo:hi
            overlap = max(0.0, min(edges[i + 1], b) - max(edges[i], a))
            total += overlap * means[i]
            largest = max(largest, overlap / W)
        end
        if largest > limit
            gaps += 1
            values[j] = NaN
        else
            values[j] = total / W
        end
    end
    return values, gaps
end

# Whether x lies between grid points one of which carries minimum-revision end weights,
# where the fitted values are interpolated as for LocalPolynomial
function _interpolate_end(state, W, x, grid_values)
    state.endpoints isa MinimumRevision && !isnothing(grid_values) || return false
    c = (x - first(state.times)) / state.Δ + 1
    isapprox(c, round(c); atol = 1e-9) && return false
    at_end(i) = (i - 1) * state.Δ < W / 2 - state.Δ / 2 - 1e-9 * W ||
                (length(state.times) - i) * state.Δ < W / 2 - state.Δ / 2 - 1e-9 * W
    return at_end(floor(Int, c)) || at_end(ceil(Int, c))
end

function _warn_gaps(gaps)
    gaps > 0 &&
        @warn("MovingAverage: $gaps value(s) set to NaN where a gap leaves the window to a " *
              "single observation (see gapfactor)",
            _group = :diagnostics,)
    return nothing
end
