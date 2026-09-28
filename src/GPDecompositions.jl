module GPDecompositions

using AbstractGPs
using Distributions
using ForwardDiff
using Interpolations: Interpolations, BSpline, Cubic, Free, OnGrid, interpolate
using LinearAlgebra: I, PosDefException, Symmetric, cholesky, diag, dot, tr
using Optim
using ParameterHandling
using Reexport
using SparseArrays: sparse, spdiagm
using Statistics
using StatsAPI
using Stheno
using Zygote

@reexport using KernelFunctions

import AbstractGPs: mean, cov, var
import StatsAPI: fit, predict, fitted, residuals
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

include("smoothers/smoothers.jl")
include("smoothers/kernels.jl")
include("smoothers/endpoints.jl")
include("smoothers/localpolynomial.jl")
include("smoothers/movingaverage.jl")
include("smoothers/loess.jl")
include("smoothers/seasonal.jl")
include("smoothers/derivative.jl")

include("classical/grid.jl")
include("classical/stages.jl")
include("classical/x11.jl")
include("classical/stl.jl")
include("classical/classical.jl")

export fit, predict, fitted, residuals
export AbstractDecomp, Component, GPDecomp, X11Decomp, STLDecomp
export AbstractSmoother, AbstractTrendSmoother, AbstractSeasonalSmoother, FittedSmoother
export LocalPolynomial, Henderson, MovingAverage, LOESS
export SeasonalMovingAverage, StableSeasonal
export AbstractEndpoints, NoEndpoints, DirectAsymmetric, MinimumRevision
export musgrave, musgrave_ic_ratio
export AbstractSmootherKernel, HendersonKernel, BiweightKernel, TricubeKernel
export EpanechnikovKernel, UniformKernel, kernel_weights

Base.@deprecate_binding Decomposition GPDecomp

end # module GPDecompositions
