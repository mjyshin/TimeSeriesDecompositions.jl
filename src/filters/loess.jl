"""
    LoessFilter <: AbstractTrendFilter

Loess (Cleveland, 1979) trend of degree `degree` with a nearest-neighbour span covering
`window` observations, computed with Loess.jl.

Fields: `window` and `degree`. Create with `fit(LoessFilter, y; window, degree=2)`.
"""
struct LoessFilter <: AbstractTrendFilter
    window::Int
    degree::Int
end

"""
    fit(LoessFilter, y; window, degree=2) -> LoessFilter

Loess trend filter spanning `window` observations (no data-dependent hyperparameters;
`y` is only validated). See [`LoessFilter`](@ref).
"""
function fit(
    ::Type{LoessFilter}, y::AbstractVector{<:Real}; window::Integer, degree::Integer=2
)
    _check_series(y)
    window ≥ 2 || throw(DomainError(window, "window must be at least 2"))
    degree ∈ (1, 2) || throw(DomainError(degree, "Loess.jl supports degree 1 or 2"))
    return LoessFilter(Int(window), Int(degree))
end

function predict(f::LoessFilter, y::AbstractVector{<:Real})
    y = _check_series(y)
    _check_length(y, f.window)
    t = collect(1.0:length(y))
    model = Loess.loess(t, y; span=f.window / length(y), degree=f.degree)
    return Loess.predict(model, t)
end

function Base.show(io::IO, f::LoessFilter)
    return print(io, "LoessFilter(window=$(f.window), degree=$(f.degree))")
end
