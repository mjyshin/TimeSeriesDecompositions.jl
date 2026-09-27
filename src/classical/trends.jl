"""
    AbstractTrendSpec

Specification of the trend filter of an [`X11Decomp`](@ref):
[`LocalPolynomialTrend`](@ref) (Henderson by default) or [`LoessTrend`](@ref). Settings
left as `nothing` are chosen when the decomposition is fitted.
"""
abstract type AbstractTrendSpec end

"""
    LocalPolynomialTrend(; window=nothing, degree=3, kernel=HendersonKernel(),
                         endpoints=MinimumRevision())

[`LocalPolynomialFilter`](@ref) trend: the Henderson filter by default. With
`window=nothing`, the length is the smallest odd integer `≥ p + 1` (at least 5) for `p`
samples per period: a trend-cycle cutoff at a fixed fraction of the seasonal cycle,
giving the 5- and 13-term filters of X-11 for quarterly and monthly data.
"""
struct LocalPolynomialTrend{K<:AbstractFilterKernel,E<:AbstractEndpoints} <:
       AbstractTrendSpec
    window::Union{Nothing,Int}
    degree::Int
    kernel::K
    endpoints::E
end

function LocalPolynomialTrend(;
    window::Union{Nothing,Integer}=nothing,
    degree::Integer=3,
    kernel::AbstractFilterKernel=HendersonKernel(),
    endpoints::AbstractEndpoints=MinimumRevision(),
)
    endpoints isa NearestNeighbours && throw(
        ArgumentError("NearestNeighbours end weights are only available for LoessTrend")
    )
    return LocalPolynomialTrend(window, Int(degree), kernel, endpoints)
end

"""
    NearestNeighbours()

End treatment of loess: at the ends of the series, fit the local polynomial to the
`window` nearest observations, as in STL. Although an [`AbstractEndpoints`](@ref), it is
only available for [`LoessTrend`](@ref). It
reproduces polynomials of the loess degree but, like [`DirectAsymmetric`](@ref), is not
optimised for revisions.
"""
struct NearestNeighbours <: AbstractEndpoints end

Base.show(io::IO, ::NearestNeighbours) = print(io, "NearestNeighbours()")

function _validate(::NearestNeighbours, _, _)
    return throw(
        ArgumentError("NearestNeighbours end weights are only available for LoessTrend")
    )
end

"""
    LoessTrend(; window=nothing, degree=2, endpoints=MinimumRevision())

Loess trend with `window` (`q`) nearest observations and local polynomials of degree
`degree`. On the regular grid of an [`X11Decomp`](@ref), its interior weights equal
those of a [`LocalPolynomialFilter`](@ref) of length `q - 2` with the
[`TricubeKernel`](@ref), which is how it is computed (so `d.θ.filters.trend` is that
filter, of length `q - 2`, while `d.θ.estimated.trend_window` reports `q`); it takes the
same `endpoints`. [`NearestNeighbours`](@ref) gives loess's own end treatment instead,
computed by a [`LoessSmoother`](@ref).

With `window=nothing`, `q` is the smallest odd integer `≥ p + 3` (at least 7) for `p`
samples per period, which matches the cutoff and noise reduction of the default
Henderson filter (e.g. `q = 15` for the 13-term Henderson filter, at degree 2).
"""
struct LoessTrend{E<:AbstractEndpoints} <: AbstractTrendSpec
    window::Union{Nothing,Int}
    degree::Int
    endpoints::E
end

function LoessTrend(;
    window::Union{Nothing,Integer}=nothing,
    degree::Integer=2,
    endpoints::AbstractEndpoints=MinimumRevision(),
)
    degree ∈ (0, 1, 2) || throw(DomainError(degree, "degree must be 0, 1 or 2"))
    return LoessTrend(window, Int(degree), endpoints)
end

_default_window(::LocalPolynomialTrend, p) = max(5, _nextodd(p + 1))
_default_window(::LoessTrend, p) = max(7, _nextodd(p + 3))

# The preliminary trend of X11Decomp: the final specification with its window and
# minimum-revision end weights of ratio 0 (loess's own ends are kept)
function _preliminary(spec::LocalPolynomialTrend, window)
    return LocalPolynomialTrend(
        window, spec.degree, spec.kernel, MinimumRevision(; ratio=0)
    )
end
function _preliminary(spec::LoessTrend, window)
    endpoints =
        spec.endpoints isa NearestNeighbours ? spec.endpoints : MinimumRevision(; ratio=0)
    return LoessTrend(window, spec.degree, endpoints)
end

# Fit the trend of the uniformly sampled series y with the given window; returns the
# fitted filter or smoother, the trend and the hyperparameters chosen from the data
function _trend(spec::LocalPolynomialTrend, y, window)
    f = _fit(
        LocalPolynomialFilter,
        y;
        window,
        degree=spec.degree,
        kernel=spec.kernel,
        endpoints=spec.endpoints,
    )
    return f, predict(f, y), f.estimated
end

function _trend(spec::LoessTrend, y, window)
    isodd(window) && window ≥ 5 ||
        throw(ArgumentError("loess trend window must be odd and ≥ 5, got $window"))
    if spec.endpoints isa NearestNeighbours
        s = _fit(LoessSmoother, y; span=window / length(y), degree=spec.degree)
        return s, predict(s), s.estimated
    end
    f = _fit(
        LocalPolynomialFilter,
        y;
        window=window - 2,
        degree=spec.degree,
        kernel=TricubeKernel(),
        endpoints=spec.endpoints,
    )
    return f, predict(f, y), f.estimated
end

function Base.show(io::IO, spec::LocalPolynomialTrend)
    return print(
        io,
        "LocalPolynomialTrend(window=$(something(spec.window, "nothing")), ",
        "degree=$(spec.degree), kernel=$(spec.kernel), endpoints=$(spec.endpoints))",
    )
end

function Base.show(io::IO, spec::LoessTrend)
    return print(
        io,
        "LoessTrend(window=$(something(spec.window, "nothing")), degree=$(spec.degree), ",
        "endpoints=$(spec.endpoints))",
    )
end
