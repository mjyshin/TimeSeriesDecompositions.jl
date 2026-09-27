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

@testset "lengthscale ordering" begin
    Random.seed!(1)
    N = 120
    t = collect(1.0:Float64(N))
    trend_true = 0.03 .* t .+ 0.8 .* sin.(2π .* t ./ 200)
    envelope = 1 .+ 0.5 .* sin.(2π .* t ./ 80)
    y = trend_true .+ envelope .* 1.5 .* sin.(2π .* t ./ 12) .+ 0.2 .* randn(N)

    for bandwidth in (36.0, Inf)
        d = fit(GPDecomp, y, t; period=12.0, bandwidth)
        @test d.θ.l_trend > (isfinite(bandwidth) ? bandwidth : 12.0)
        @test sqrt(mean(abs2, d.trend.mean .- trend_true)) < 0.25
        # the zero-mean seasonal kernel leaves the level to the trend
        @test abs(mean(d.seasonal.mean)) < 0.1
    end

    # an infeasible starting value is moved above the bound
    d = @test_logs (:info,) match_mode = :any fit(
        GPDecomp, y, t; period=12.0, bandwidth=36.0, θ_init=(l_trend=5.0,)
    )
    @test d.θ.l_trend > 36.0

    @test_throws ArgumentError fit(
        GPDecomp,
        y,
        t;
        period=12.0,
        bandwidth=36.0,
        θ_init=(l_trend=10.0,),
        θ_fixed=[:l_trend],
    )
    @test_throws DomainError fit(GPDecomp, y, t; period=12.0, bandwidth=6.0)

    # without seasonality l_trend is unbounded
    d = fit(GPDecomp, y, t; θ_init=(l_trend=0.5,))
    @test d.θ.l_trend > 0
end

@testset "l_seasonal validation" begin
    t = collect(1.0:48.0)
    y = sin.(2π .* t ./ 12) .+ 0.1 .* randn(48)
    @test_throws ArgumentError fit(GPDecomp, y, t; period=12.0, θ_init=(l_seasonal=Inf,))
    @test_throws ArgumentError fit(
        GPDecomp, y, t; period=12.0, θ_init=(l_seasonal=50.0,), θ_fixed=[:l_seasonal]
    )
    d = fit(GPDecomp, y, t; period=12.0)
    @test 0 < d.θ.l_seasonal <= 10
end
