module GPEstimators

using AbstractGPs
using Distributions
using FiniteDifferences
using KernelFunctions
using Optim
using ParameterHandling
using Statistics
using Stheno
using Zygote

import AbstractGPs: mean, cov, var
using AbstractGPs: AbstractGP, PosteriorGP
using Stheno: DerivedGP

include("defaults.jl")

include("models.jl")

include("derivatives.jl")

include("optimize.jl")

include("helpers.jl")
export comp_decomp

include("decompose.jl")
export decompose

include("misc.jl")

end # module GPEstimators
