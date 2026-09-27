"""
    MovingAverage <: AbstractTrendFilter

Centred simple moving average of length `window`. An even `window = m` gives the
centred `2 × m` average (weights `[1/2, 1, …, 1, 1/2] / m`, length `m + 1`), e.g. the
`2 × 12` average of X-11 for monthly data. At the ends, the weights falling outside the
series are dropped and the rest renormalised.

Fields: `window` and `weights`. Create with `fit(MovingAverage, y; window)`.
"""
struct MovingAverage <: AbstractTrendFilter
    window::Int
    weights::Vector{Float64}
end

"""
    fit(MovingAverage, y; window) -> MovingAverage

Moving average of length `window` (no data-dependent hyperparameters; `y` is only
validated). See [`MovingAverage`](@ref).
"""
function fit(::Type{MovingAverage}, y::AbstractVector{<:Real}; window::Integer)
    _check_series(y)
    window ≥ 1 || throw(DomainError(window, "window must be positive"))
    w = isodd(window) ? ones(window) : [0.5; ones(window - 1); 0.5]
    return MovingAverage(Int(window), w ./ window)
end

function predict(f::MovingAverage, y::AbstractVector{<:Real})
    return _truncated_filter(f.weights, _check_series(y))
end

Base.show(io::IO, f::MovingAverage) = print(io, "MovingAverage(window=$(f.window))")
