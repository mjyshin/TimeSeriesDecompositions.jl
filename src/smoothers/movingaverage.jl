"""
    MovingAverage(; bandwidth=nothing, endpoints=NoEndpoints(), gapfactor=3)
        <: AbstractTrendSmoother

Moving average over a window of length `bandwidth` (in units of `t`, twice the radius),
defined for any times as the Gasser–Müller box average: the average over
`[x - bandwidth/2, x + bandwidth/2]` of the piecewise-constant interpolant that gives
each observation its Voronoi cell (the interval between the midpoints to its neighbours;
the outer cells extend half a spacing beyond the data, and tied times share one cell).
Observation `i` thus has weight `|cellᵢ ∩ window| / bandwidth`.

On a regular grid of spacing `Δ` with `bandwidth = mΔ` it is exactly the centred `2 × m`
average for even `m` (weights `[½, 1, …, 1, ½] / m` on the `m + 1` samples within
`bandwidth / 2`, the outer two half covered) and the `m`-term moving average for odd `m`,
which annihilates any fixed seasonal pattern of period `m`; for irregular `t` it
approximates the time integral over the window, so it approximately annihilates it.
It reproduces constants, but lines only where the observations are symmetric about `x`.

- `bandwidth`: window length in units of `t` (required; inside a decomposition,
  `nothing` gives the period).
- `endpoints`: [`NoEndpoints`](@ref) (default) gives `NaN` where the window extends beyond
  the outer cells, as in R's `decompose`; on a regular grid,
  [`MinimumRevision`](@ref)`(degree=0)` or [`DirectAsymmetric`](@ref) (the truncated,
  renormalised average) fill those points.
- `gapfactor`: the average is `NaN`, with a warning, where one observation's cell covers
  more than `gapfactor` typical spacings of the window, i.e. where a gap in the data
  leaves the window to a single observation. The typical spacing is the gap containing a
  random time (the median of the gaps weighted by their length), so bursts of closely
  spaced observations do not shrink it. When the bandwidth spans at most `gapfactor`
  spacings the rule cannot apply, and within a gap the average is that of the nearest
  observation.
"""
struct MovingAverage{E<:AbstractEndpoints} <: AbstractTrendSmoother
    bandwidth::Union{Nothing,Float64}
    endpoints::E
    gapfactor::Float64
end

function MovingAverage(;
    bandwidth::Union{Nothing,Real}=nothing,
    endpoints::AbstractEndpoints=NoEndpoints(),
    gapfactor::Real=3,
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
        edges=_cell_edges(times),
        limit=model.gapfactor * spacing / W,
        kernel=grid,
        endpoints,
        degree=0,
        Δ=isnothing(grid) ? nothing : _spacing(t),
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

# Distinct sorted times and the mean of the observations at each
function _pool_ties(t, y)
    times = unique(t)
    means = [mean(y[searchsortedfirst(t, u):searchsortedlast(t, u)]) for u in times]
    return times, means
end

# Median of the gaps between times, weighted by their length: the gap containing a
# random time in the observed range
function _typical_spacing(times)
    gaps = sort(diff(times))
    cumulative = cumsum(gaps)
    return gaps[searchsortedfirst(cumulative, last(cumulative) / 2)]
end

# Voronoi cell boundaries: midpoints between neighbours, with the outer cells extending
# half a spacing beyond the data
function _cell_edges(times)
    mids = (times[1:(end - 1)] .+ times[2:end]) ./ 2
    return [
        times[1] - (times[2] - times[1]) / 2;
        mids;
        times[end] + (times[end] - times[end - 1]) / 2
    ]
end

# On a regular grid, the box average at offset d (in samples) from x is the local
# constant fit with weights |[d - ½, d + ½] ∩ [-b, b]| for b = bandwidth / 2Δ
function _box_kernel(m::Real)
    b = m / 2
    return _GridKernel(d -> clamp(b + 0.5 - abs(d), 0.0, 1.0), ceil(Int, b + 0.5) - 1)
end

# Box averages at the times xs, and the number of NaNs caused by gaps; near the ends of a
# grid with end weights, off-grid values interpolate the fitted values at the grid
function _box_average(state, W, y, xs, grid_values=nothing)
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
                isapprox(c, round(c); atol=1e-9) && (c = round(c))
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

# Whether x lies strictly between two grid points one of which carries minimum-revision
# end weights, where the fitted values are interpolated (as for LocalPolynomial) so that
# the curve is continuous where the box average hands over to the end weights
function _interpolate_end(state, W, x, grid_values)
    state.endpoints isa MinimumRevision && !isnothing(grid_values) || return false
    c = (x - first(state.times)) / state.Δ + 1
    isapprox(c, round(c); atol=1e-9) && return false
    at_end(i) =
        (i - 1) * state.Δ < W / 2 - state.Δ / 2 - 1e-9 * W ||
        (length(state.times) - i) * state.Δ < W / 2 - state.Δ / 2 - 1e-9 * W
    return at_end(floor(Int, c)) || at_end(ceil(Int, c))
end

function _warn_gaps(gaps)
    gaps > 0 && @warn(
        "MovingAverage: $gaps value(s) set to NaN where a gap leaves the window to a " *
            "single observation (see gapfactor)",
        _group = :diagnostics,
    )
    return nothing
end
