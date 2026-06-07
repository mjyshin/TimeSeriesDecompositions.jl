module GPDecompositions

using AbstractGPs
using Distributions
using ForwardDiff
using KernelFunctions
using Optim
using ParameterHandling
using LinearAlgebra: diag
using Statistics
using StatsAPI
using Stheno
using Zygote

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

export Decomposition, Component
export fit, predict

end # module GPDecompositions
