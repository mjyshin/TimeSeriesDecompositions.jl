# Derivative of a uniformly sampled series per sample: slope of the cubic interpolating
# spline in the interior, second-order one-sided differences at the two ends
function _derivative(x::AbstractVector{<:Real})
    N = length(x)
    N ≥ 3 || throw(ArgumentError("need at least 3 points to differentiate"))
    spline = cubic_spline_interpolation(1:N, x)
    dx = [Interpolations.gradient(spline, t)[1] for t in 1:N]
    dx[1] = (-3x[1] + 4x[2] - x[3]) / 2
    dx[N] = (3x[N] - 4x[N - 1] + x[N - 2]) / 2
    return dx
end
