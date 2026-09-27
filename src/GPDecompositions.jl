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

include("types.jl")
include("utils.jl")
include("params.jl")
include("kernels.jl")
include("derivatives.jl")
include("fit.jl")
include("predict.jl")

# Classical (filter-based) methods, imported verbatim from RealTimeEstimators.jl
include("RealTimeEstimators/RealTimeEstimators.jl")

export Decomposition, Component
export fit, predict

end # module GPDecompositions
