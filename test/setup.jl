# Shared fixtures. Random data come from StableRNGs, whose streams stay the same across
# Julia versions (Random's default may change in a minor release), so thresholds tuned to
# the draws hold on every supported version.

using StableRNGs: StableRNG

# A regular series of 50 points with period 12 (the API tests)
const _t = collect(1.0:50.0)
const _y = sin.(2π .* _t ./ 12) .+ 0.01 .* _t .+ 0.1 .* randn(StableRNG(1), 50)

# Ten years of monthly data with a linear trend (the X11 and STL tests)
const _tc = collect(2000 .+ (0:119) ./ 12)
const _yc = sin.(2π .* _tc) .+ 0.2 .* (_tc .- 2000) .+ 0.1 .* randn(StableRNG(2), 120)

# A noisier series of 120 points with period 12 (the smoother tests)
const _ys = [sin(2π * t / 12) + 0.02t + 0.3 * x
             for (t, x) in zip(1:120, randn(StableRNG(3), 120))]
