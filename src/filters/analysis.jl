##############################
###                          #
##    ANALYSIS FUNCTIONS    ##
#                          ###
##############################

#== Derivatives ==#

𝒟(f::Interpolations.Extrapolation) = x -> Interpolationsgradient(f, x)[1]
𝒟(f::Interpolations.Extrapolation, x::Real) = gradient(f, x)[1]
𝒟(f::Interpolations.Extrapolation, x::AbstractVector{<:Real}) = 𝒟.(Ref(f), x)
𝒟²(f::Interpolations.Extrapolation) = x -> hessian(f, x)[1]
𝒟²(f::Interpolations.Extrapolation, x::Real) = hessian(f, x)[1]
𝒟²(f::Interpolations.Extrapolation, x::AbstractVector{<:Real}) = 𝒟².(Ref(f), x)

"""
    derivative(f::Vector{<:Real}; second::Bool=false)

Evenly-sampled derivative(s) df/dx (and d²f/dx²) of even samples of f(x).
"""
function derivative(f::Vector{<:Real}; second::Bool=false)
    x = 1:length(f)
    f̂ = cubic_spline_interpolation(x, f)    # interpolation f̂(x) ≈ f(x)

    df = [
        (-3f[1] + 4f[2] - f[3])/2;
        𝒟(f̂, x[2:end-1]);
        (3f[end] - 4f[end-1] + f[end-2])/2
    ]    # compute slope at indices
    if second
        d2f = [
            (-3df[1] + 4df[2] - df[3])/2;
            𝒟²(f̂, x[2:end-1]);
            (3df[end] - 4df[end-1] + df[end-2])/2
        ]    # compute curvature at indices
        return df, d2f
    end

    return df
end


#== Growth rates ==#

"""
    rates(x::Vector{<:Real}, h::Int; Ft::Function=mrf, kwargs...)

Compute smoothed growth rates (rₜ) from smoothed series (xₜ) s.t. dx/dt ≈ r(t) or dx/dt ≈ r(t)x(t).
"""
function rates(x::Vector{<:Real}, h::Int=0; Ft::Function=mrf, logscale=false, dt::Real=1, kwargs...)
    if logscale x = log.(x) end
    dx = derivative(x)
    r = Ft(dx, h; kwargs...)/dt

    return r
end
