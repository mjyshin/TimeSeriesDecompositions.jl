module GPEstimators

using AbstractGPs
using Distributions
using ForwardDiff
using KernelFunctions
using Optim
using ParameterHandling
using LinearAlgebra: diag
using Statistics
using Stheno
using Zygote

import AbstractGPs: mean, cov, var
using AbstractGPs: AbstractGP, PosteriorGP
using KernelFunctions: kernelmatrix
using ParameterHandling: positive, fixed
using Stheno: DerivedGP

include("types.jl")
include("utils.jl")
include("params.jl")
include("kernels.jl")
include("derivatives.jl")
include("fit.jl")
include("predict.jl")

export Decomposition, ComponentEstimate
export decompose, predict

end # module GPEstimators
