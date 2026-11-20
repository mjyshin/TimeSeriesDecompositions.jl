module GPDecompositions

## Dependencies ##

using ADTypes: AbstractADType, AutoZygote
using AbstractGPs: GP, PosteriorGP, logpdf, posterior
using Distributions: Chisq, Normal, quantile
using HypothesisTests: LjungBoxTest, pvalue
using Interpolations: Interpolations, BSpline, Cubic, Free, OnGrid, interpolate
using LinearAlgebra: Cholesky, I, PosDefException, dot, tr
using Optim: Optim, Brent, LBFGS, optimize
using ParameterHandling: ParameterHandling, bounded, fixed, positive, value_flatten
using Reexport: @reexport
using SparseArrays: sparse, spdiagm
using Statistics: mean, median, var
using StatsBase: autocor, mean_and_std
import StatsAPI: fit, predict, fitted, residuals
using Zygote: Zygote    # loaded for the default `autodiff = AutoZygote()`

@reexport using KernelFunctions
using KernelFunctions: KernelFunctions, Kernel, Matern52Kernel, PeriodicTransform, ScaleTransform,
                       ScaledKernel,
                       TransformedKernel, WhiteKernel, kernelmatrix, kernelmatrix_diag,
                       with_lengthscale

## Source ##

include("common.jl")

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

## Exports ##

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
