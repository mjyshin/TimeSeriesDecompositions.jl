"""
    Filters

Classical trend and seasonal filters (moving averages, Henderson with DAF/MRF end
weights, loess, stable and S(n,m) seasonal filters), imported verbatim from
RealTimeEstimators.jl. These are the building blocks of `X11Decomp` and `STLDecomp`.

Internal: nothing is exported; call as `Filters.x11`, `Filters.mrf`, etc.
"""
module Filters

using Interpolations
using LinearAlgebra
using Loess
using Statistics
using StatsBase

import Interpolations: gradient, hessian

include("trend.jl")      # sma, henderson, daf, mrf, loess
include("seasonal.jl")   # ssf, snm
include("decompose.jl")  # deseasonalize, decompose, x11
include("analysis.jl")   # derivative, rates

end # module Filters
