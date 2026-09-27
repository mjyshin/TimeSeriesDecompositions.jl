module GPDecompositions

using AbstractGPs
using Distributions
using ForwardDiff
using Optim
using ParameterHandling
using LinearAlgebra: diag
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

# Shared types and helpers
include("common.jl")

# Gaussian-process decomposition
include("gp/types.jl")
include("gp/utils.jl")
include("gp/params.jl")
include("gp/kernels.jl")
include("gp/derivatives.jl")
include("gp/fit.jl")
include("gp/predict.jl")

# Classical filter-based decompositions
include("filters/Filters.jl")
include("x11.jl")
include("stl.jl")
include("filterdecomps.jl")

export AbstractDecomp, Component
export GPDecomp, X11Decomp, STLDecomp
export fit, predict

Base.@deprecate_binding Decomposition GPDecomp

end # module GPDecompositions
