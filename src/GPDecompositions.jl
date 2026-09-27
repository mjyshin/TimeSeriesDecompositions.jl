module GPDecompositions

using AbstractGPs
using Distributions
using ForwardDiff
using Interpolations: Interpolations, cubic_spline_interpolation
using LinearAlgebra: I, diag, dot
using Loess: Loess
using Optim
using ParameterHandling
using Reexport
using Statistics
using StatsAPI
using Stheno
using Zygote

@reexport using KernelFunctions

import AbstractGPs: mean, cov, var
import StatsAPI: fit, predict
using AbstractGPs: AbstractGP, PosteriorGP
using KernelFunctions: kernelmatrix
using ParameterHandling: positive, fixed

include("common.jl")

include("gp/types.jl")
include("gp/utils.jl")
include("gp/params.jl")
include("gp/kernels.jl")
include("gp/derivatives.jl")
include("gp/fit.jl")
include("gp/predict.jl")

include("filters/filters.jl")
include("filters/kernels.jl")
include("filters/localpolynomial.jl")
include("filters/movingaverage.jl")
include("filters/loess.jl")
include("filters/seasonal.jl")
include("filters/derivative.jl")

include("classical/grid.jl")
include("classical/x11.jl")
include("classical/stl.jl")
include("classical/classical.jl")

export fit, predict
export AbstractDecomp, Component, GPDecomp, X11Decomp, STLDecomp
export AbstractFilter, AbstractTrendFilter, AbstractSeasonalFilter
export LocalPolynomialFilter, MovingAverage, LoessFilter
export StableSeasonalFilter, SeasonalMovingAverage
export AbstractEndpoints, DirectAsymmetric, MinimumRevision, musgrave, musgrave_ic_ratio
export filter_kernel

Base.@deprecate_binding Decomposition GPDecomp

end # module GPDecompositions
