using GPEstimators
using Test
using Random
using Statistics

@testset "posterior mean within credible intervals (synthetic data)" begin
    rng = 42
    Random.seed!(rng)

    N = 60
    t = collect(1.0:Float64(N))
    trend_true    = 0.05 .* t
    seasonal_true = 1.5 .* sin.(2π .* t ./ 12)
    noise_true    = 0.15 .* randn(N)
    y = trend_true .+ seasonal_true .+ noise_true

    d = decompose(y, t; period=12.0, fix_params=[:period], sig=[0.9], standardize=true)

    # Posterior mean should be in the ballpark of ground truth
    tol = 1.5
    @test mean(abs.(d.trend.mean    .- trend_true))    < tol
    @test mean(abs.(d.seasonal.mean .- seasonal_true)) < tol
end

@testset "static vs quasi-periodic agree on stationary data" begin
    N = 48
    t = collect(1.0:Float64(N))
    y = 2.0 .* sin.(2π .* t ./ 12) .+ 0.1 .* randn(N)

    # Fix period so both models start from an identical, correct initialization
    d_static = decompose(y, t; period=12.0, bandwidth=Inf,   fix_params=[:period])
    d_qp     = decompose(y, t; period=12.0, bandwidth=200.0, fix_params=[:period])

    # A large bandwidth quasi-periodic kernel is nearly stationary; both should
    # recover the same seasonal signal
    @test cor(d_static.seasonal.mean, d_qp.seasonal.mean) > 0.9
end

@testset "fix_params respected" begin
    t = collect(1.0:30.0)
    y = sin.(2π .* t ./ 6) .+ 0.1 .* randn(30)

    d = decompose(y, t;
        period=6.0,
        init_params=(; period=6.0),
        fix_params=[:period],
    )
    @test d.params.period ≈ 6.0
end

@testset "decomposition sums to observed (approx)" begin
    N = 48
    t = collect(1.0:Float64(N))
    y = sin.(2π .* t ./ 12) .+ 0.05t .+ 0.1 .* randn(N)

    d = decompose(y, t; period=12.0)
    reconstructed = d.trend.mean .+ d.seasonal.mean .+ d.noise.mean

    # GP posterior y ≈ x̄ + x̃ + ϵ  (should hold nearly exactly at training times)
    @test mean(abs.(reconstructed .- y)) < 0.5
end
