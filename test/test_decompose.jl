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

    d_static = fit(GPDecomp, y, t; period=12.0, θ_fixed=(bandwidth=Inf,))
    d_qp = fit(GPDecomp, y, t; period=12.0, θ_fixed=(bandwidth=200.0,))

    @test cor(d_static.seasonal.mean, d_qp.seasonal.mean) > 0.9
end

@testset "θ_fixed respected" begin
    t = collect(1.0:30.0)
    y = sin.(2π .* t ./ 6) .+ 0.1 .* randn(30)

    d = fit(GPDecomp, y, t; period=6.0, θ_fixed=(σ_noise=0.1,))
    @test d.θ.period ≈ 6.0
    @test d.θ.σ_noise ≈ 0.1
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
        d = fit(GPDecomp, y, t; period=12.0, θ_fixed=(; bandwidth))
        @test d.θ.l_trend > 12.0
        @test sqrt(mean(abs2, d.trend.mean .- trend_true)) < 0.25
        # the zero-mean seasonal kernel leaves the level to the trend
        @test abs(mean(d.seasonal.mean)) < 0.1
    end

    # an infeasible starting value is moved above the bound
    d = @test_logs (:info,) match_mode = :any fit(
        GPDecomp, y, t; period=12.0, θ_init=(l_trend=5.0,), θ_fixed=(bandwidth=36.0,)
    )
    @test d.θ.l_trend > 12.0

    @test_throws ArgumentError fit(
        GPDecomp, y, t; period=12.0, θ_fixed=(bandwidth=36.0, l_trend=10.0)
    )
    @test_throws DomainError fit(GPDecomp, y, t; period=12.0, θ_fixed=(bandwidth=6.0,))

    # without seasonality l_trend is unbounded
    d = fit(GPDecomp, y, t; θ_init=(l_trend=0.5,))
    @test d.θ.l_trend > 0
end

@testset "l_seasonal validation" begin
    t = collect(1.0:48.0)
    y = sin.(2π .* t ./ 12) .+ 0.1 .* randn(48)
    @test_throws ArgumentError fit(GPDecomp, y, t; period=12.0, θ_init=(l_seasonal=Inf,))
    @test_throws ArgumentError fit(GPDecomp, y, t; period=12.0, θ_fixed=(l_seasonal=50.0,))
    d = fit(GPDecomp, y, t; period=12.0)
    @test 0 < d.θ.l_seasonal <= 10
end

@testset "bandwidth estimation" begin
    Random.seed!(3)
    t = collect(1.0:120.0)
    envelope = 1 .+ 0.6 .* sin.(2π .* t ./ 60)
    evolving = 0.02 .* t .+ envelope .* sin.(2π .* t ./ 12) .+ 0.2 .* randn(120)
    static = 0.02 .* t .+ sin.(2π .* t ./ 12) .+ 0.2 .* randn(120)

    d = fit(GPDecomp, evolving, t; period=12.0)
    test = d._fit_info.fit.bandwidth_test
    @test !test.static_selected
    @test test.lr_static > test.critical
    @test 12 < d.θ.bandwidth < 120
    @test :bandwidth ∈ d._fit_info.fit.free
    @test test.loglik_static < d._fit_info.fit.loglik

    d = fit(GPDecomp, static, t; period=12.0)
    @test d._fit_info.fit.bandwidth_test.static_selected
    @test d.θ.bandwidth == Inf
    @test :bandwidth ∈ d._fit_info.fit.free

    # θ_fixed fixes it; θ_init is an extra start; no seasonal, no estimation
    d = fit(GPDecomp, evolving, t; θ_fixed=(period=12.0, bandwidth=48.0))
    @test d.θ.bandwidth ≈ 48.0
    @test isnothing(d._fit_info.fit.bandwidth_test)
    @test :bandwidth ∉ d._fit_info.fit.free
    d = fit(GPDecomp, evolving, t; period=12.0, θ_init=(bandwidth=48.0,))
    @test !isnothing(d._fit_info.fit.bandwidth_test)
    @test 12 < d.θ.bandwidth < 120
    @test isnothing(fit(GPDecomp, evolving, t)._fit_info.fit.bandwidth_test)
    @test_throws DomainError fit(
        GPDecomp, evolving, t; θ_init=(period=12.0, bandwidth=6.0)
    )
end

@testset "innovation whiteness" begin
    Random.seed!(4)
    t = collect(1.0:150.0)
    white = fit(GPDecomp, 0.02 .* t .+ 0.3 .* randn(150), t)
    @test white._fit_info.fit.whiteness.pvalue > 0.01
    @test white._fit_info.fit.whiteness.lags == 10
    # AR(1) noise with strong positive correlation is not white
    ϵ = zeros(150)
    for i in 2:150
        ϵ[i] = 0.8ϵ[i - 1] + 0.3randn()
    end
    y = 0.5 .* sin.(2π .* t ./ 150) .+ ϵ
    w = GPDecompositions._innovation_whiteness(
        GPDecompositions._build_gp(
            (σ_trend=1.0, l_trend=500.0, σ_seasonal=0.0, l_seasonal=Inf, period=Inf,
                ρ=0.0, σ_noise=1.0),
        )(GPDecompositions.GPPPInput(:y, t), 1e-6),
        (y .- mean(y)) ./ std(y),
        t,
        Inf,
    )
    @test w.pvalue < 0.01
    @test w.max_lag == 1 && w.max_acf > 0
end
