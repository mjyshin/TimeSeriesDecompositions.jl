module RealTimeEstimators

using Interpolations
using LinearAlgebra
using Loess
using Statistics
using StatsBase

import Interpolations: gradient, hessian

include("trend.jl")
export sma
export henderson
export daf
export mrf
export loess

include("seasonal.jl")
export ssf
export snm

include("decompose.jl")
export deseasonalize
export decompose
export x11

include("analysis.jl")
export derivative
export rates

end # module RealTimeEstimators
