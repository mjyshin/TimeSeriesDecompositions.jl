# Not-a-knot cubic spline interpolating a uniformly sampled series, indexed by sample
function _spline(x::AbstractVector{<:Real})
    return interpolate(Float64.(x), BSpline(Cubic(Free(OnGrid()))))
end

# Derivative per sample of a uniformly sampled series: the slope of its not-a-knot
# interpolating cubic spline
function _derivative(x::AbstractVector{<:Real})
    N = length(x)
    N ≥ 3 || throw(ArgumentError("need at least 3 points to differentiate"))
    spline = _spline(x)
    return [Interpolations.gradient(spline, k)[1] for k in 1:N]
end
