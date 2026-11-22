module TimeSeriesDecompositions

## Dependencies ##

using AbstractGPs: GP, PosteriorGP, logpdf, posterior
using Distributions: Chisq, Normal, quantile
using ForwardDiff: ForwardDiff
using HypothesisTests: LjungBoxTest, pvalue
using Interpolations: Interpolations, BSpline, Cubic, Free, OnGrid, interpolate
using Logging: Logging
using LoggingExtras: MinLevelLogger
using LinearAlgebra: Cholesky, I, PosDefException, Symmetric, cholesky!, diag, diagind, dot,
                     issuccess, logdet, mul!, tr
using Optim: Optim, Brent, LBFGS, OnceDifferentiable, optimize
using RecipesBase: RecipesBase, @recipe, @series
using Reexport: @reexport
using SparseArrays: sparse, spdiagm
using Statistics: mean, median, var
using StatsBase: autocor, mean_and_std
import StatsAPI: fit, predict, fitted, residuals

@reexport using KernelFunctions
using KernelFunctions: KernelFunctions, Kernel, Matern52Kernel, PeriodicTransform,
                       ScaleTransform,
                       ScaledKernel,
                       TransformedKernel, kernelmatrix, kernelmatrix_diag,
                       with_lengthscale

## Source ##

include("common.jl")
include("logging.jl")

include("gp/types.jl")
include("gp/params.jl")
include("gp/kernels.jl")
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

include("plotting.jl")

## Exports ##

export fit, predict, fitted, residuals
export quietly, @quietly
export AbstractDecompositionModel, GaussianProcess, X11, STL, Decomposition, Component
export AbstractSmoother, AbstractTrendSmoother, AbstractSeasonalSmoother, FittedSmoother
export LocalPolynomial, Henderson, MovingAverage, LOESS
export SeasonalMovingAverage, StableSeasonal
export AbstractEndpoints, NoEndpoints, DirectAsymmetric, MinimumRevision
export musgrave, musgrave_ic_ratio
export AbstractSmootherKernel, HendersonKernel, BiweightKernel, TricubeKernel
export EpanechnikovKernel, UniformKernel, kernel_weights

end # module TimeSeriesDecompositions
