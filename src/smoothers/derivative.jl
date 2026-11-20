# Not-a-knot cubic spline (Interpolations' `Free`) through a uniformly sampled series
function _spline(x::AbstractVector{<:Real})
    return interpolate(Float64.(x), BSpline(Cubic(Free(OnGrid()))))
end

# Derivative per sample of the spline
function _derivative(x::AbstractVector{<:Real})
    N = length(x)
    N ≥ 3 || throw(ArgumentError("need at least 3 points to differentiate"))
    spline = _spline(x)
    return [Interpolations.gradient(spline, k)[1] for k in 1:N]
end
