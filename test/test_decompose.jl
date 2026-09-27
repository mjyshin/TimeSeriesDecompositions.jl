using GPDecompositions
using Test
using Random
using Statistics

@testset "posterior mean within credible intervals (synthetic data)" begin
    rng = 42
    Random.seed!(rng)

    N = 60
    t = collect(1.0:Float64(N))
    trend_true = 0.05 .* t
    seasonal_true = 1.5 .* sin.(2π .* t ./ 12)
    noise_true = 0.15 .* randn(N)
    y = trend_true .+ seasonal_true .+ noise_true

    d = fit(GPDecomp, y, t; period=12.0, sig=[0.9])

    tol = 1.5
    @test mean(abs.(d.trend.mean .- trend_true)) < tol
    @test mean(abs.(d.seasonal.mean .- seasonal_true)) < tol
end

@testset "static vs quasi-periodic agree on stationary data" begin
    N = 48
    t = collect(1.0:Float64(N))
    y = 2.0 .* sin.(2π .* t ./ 12) .+ 0.1 .* randn(N)

    d_static = fit(GPDecomp, y, t; period=12.0, bandwidth=Inf)
    d_qp = fit(GPDecomp, y, t; period=12.0, bandwidth=200.0)

    @test cor(d_static.seasonal.mean, d_qp.seasonal.mean) > 0.9
end

@testset "θ_fixed respected" begin
    t = collect(1.0:30.0)
    y = sin.(2π .* t ./ 6) .+ 0.1 .* randn(30)

    d = fit(GPDecomp, y, t; period=6.0, θ_fixed=[:σ_noise])
    @test d.θ.period ≈ 6.0
end

@testset "decomposition sums to observed (approx)" begin
    N = 48
    t = collect(1.0:Float64(N))
    y = sin.(2π .* t ./ 12) .+ 0.05t .+ 0.1 .* randn(N)

    d = fit(GPDecomp, y, t; period=12.0)
    reconstructed = d.trend.mean .+ d.seasonal.mean .+ d.noise.mean

    @test mean(abs.(reconstructed .- y)) < 0.5
end
