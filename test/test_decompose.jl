using GPDecompositions
using Test
using Random
using Statistics
using LinearAlgebra: I, Symmetric, cholesky
using GPDecompositions: GP, logpdf

@testset "posterior mean within credible intervals (synthetic data)" begin
    rng = 42
    Random.seed!(rng)

    N = 60
    t = collect(1.0:Float64(N))
    trend_true = 0.05 .* t
    seasonal_true = 1.5 .* sin.(2π .* t ./ 12)
    noise_true = 0.15 .* randn(N)
    y = trend_true .+ seasonal_true .+ noise_true

    d = fit(GaussianProcess, y, t; period = 12.0)

    tol = 1.5
    @test mean(abs.(d.trend.mean .- trend_true)) < tol
    @test mean(abs.(d.seasonal.mean .- seasonal_true)) < tol
end

@testset "static vs quasi-periodic agree on stationary data" begin
    N = 48
    t = collect(1.0:Float64(N))
    y = 2.0 .* sin.(2π .* t ./ 12) .+ 0.1 .* randn(N)

    d_static = fit(GaussianProcess, y, t; period = 12.0, θ_fixed = (bandwidth = Inf,))
    d_qp = fit(GaussianProcess, y, t; period = 12.0, θ_fixed = (bandwidth = 200.0,))

    @test cor(d_static.seasonal.mean, d_qp.seasonal.mean) > 0.9
end

@testset "θ_fixed respected" begin
    # seed 31: the static fit's l_trend ends on its bound, the period, and warm-starts a
    # bandwidth fit from there
    Random.seed!(31)
    t = collect(1.0:30.0)
    y = sin.(2π .* t ./ 6) .+ 0.1 .* randn(30)

    d = fit(GaussianProcess, y, t; period = 6.0, θ_fixed = (σ_noise = 0.1,))
    @test d.period ≈ 6.0
    @test d.θ.σ_noise ≈ 0.1
end

@testset "decomposition sums to observed (approx)" begin
    N = 48
    t = collect(1.0:Float64(N))
    y = sin.(2π .* t ./ 12) .+ 0.05t .+ 0.1 .* randn(N)

    d = fit(GaussianProcess, y, t; period = 12.0)
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
        d = fit(GaussianProcess, y, t; period = 12.0, θ_fixed = (; bandwidth))
        @test d.θ.l_trend > 12.0
        @test sqrt(mean(abs2, d.trend.mean .- trend_true)) < 0.25
        # the zero-mean seasonal kernel leaves the level to the trend
        @test abs(mean(d.seasonal.mean)) < 0.1
    end

    # an infeasible starting value is moved above the bound
    d = @test_logs (:info,) match_mode = :any fit(
        GaussianProcess, y, t; period = 12.0, θ_init = (l_trend = 5.0,),
        θ_fixed = (bandwidth = 36.0,)
    )
    @test d.θ.l_trend > 12.0

    @test_throws ArgumentError fit(
        GaussianProcess, y, t; period = 12.0, θ_fixed = (bandwidth = 36.0, l_trend = 10.0)
    )
    @test_throws DomainError fit(
        GaussianProcess, y, t; period = 12.0, θ_fixed = (bandwidth = 6.0,))

    # without seasonality l_trend is unbounded
    d = fit(GaussianProcess, y, t; θ_init = (l_trend = 0.5,))
    @test d.θ.l_trend > 0
end

@testset "l_seasonal validation" begin
    t = collect(1.0:48.0)
    y = sin.(2π .* t ./ 12) .+ 0.1 .* randn(48)
    @test_throws ArgumentError fit(
        GaussianProcess, y, t; period = 12.0, θ_init = (l_seasonal = Inf,))
    @test_throws ArgumentError fit(
        GaussianProcess, y, t; period = 12.0, θ_fixed = (l_seasonal = 50.0,))
    d = fit(GaussianProcess, y, t; period = 12.0)
    @test 0 < d.θ.l_seasonal <= 10
end

@testset "bandwidth estimation" begin
    Random.seed!(3)
    t = collect(1.0:120.0)
    envelope = 1 .+ 0.6 .* sin.(2π .* t ./ 60)
    evolving = 0.02 .* t .+ envelope .* sin.(2π .* t ./ 12) .+ 0.2 .* randn(120)
    static = 0.02 .* t .+ sin.(2π .* t ./ 12) .+ 0.2 .* randn(120)

    d = fit(GaussianProcess, evolving, t; period = 12.0)
    test = d._fit_info.fit.bandwidth_test
    @test !test.static_selected
    @test test.lr_static > test.critical
    @test 12 < d.θ.bandwidth < 120
    @test :bandwidth ∈ d._fit_info.fit.free
    @test test.loglik_static < d._fit_info.fit.loglik

    d = fit(GaussianProcess, static, t; period = 12.0)
    @test d._fit_info.fit.bandwidth_test.static_selected
    @test d.θ.bandwidth == Inf
    @test :bandwidth ∈ d._fit_info.fit.free

    # θ_fixed fixes it; θ_init is an extra start; no seasonal, no estimation
    d = fit(GaussianProcess, evolving, t; period = 12.0, θ_fixed = (bandwidth = 48.0,))
    @test d.θ.bandwidth ≈ 48.0
    @test isnothing(d._fit_info.fit.bandwidth_test)
    @test :bandwidth ∉ d._fit_info.fit.free
    d = fit(GaussianProcess, evolving, t; period = 12.0, θ_init = (bandwidth = 48.0,))
    @test !isnothing(d._fit_info.fit.bandwidth_test)
    @test 12 < d.θ.bandwidth < 120
    @test isnothing(fit(GaussianProcess, evolving, t)._fit_info.fit.bandwidth_test)
    @test_throws DomainError fit(
        GaussianProcess, evolving, t; period = 12.0, θ_init = (bandwidth = 6.0,)
    )
end

@testset "innovation whiteness" begin
    Random.seed!(4)
    t = collect(1.0:150.0)
    white = fit(GaussianProcess, 0.02 .* t .+ 0.3 .* randn(150), t)
    @test white._fit_info.fit.whiteness.pvalue > 0.01
    @test white._fit_info.fit.whiteness.lags == 10
    # AR(1) noise with strong positive correlation is not white
    ϵ = zeros(150)
    for i in 2:150
        ϵ[i] = 0.8ϵ[i - 1] + 0.3randn()
    end
    y = 0.5 .* sin.(2π .* t ./ 150) .+ ϵ
    θ = (σ_trend = 1.0, l_trend = 500.0, σ_seasonal = 0.0, l_seasonal = Inf,
        period = Inf, ρ = 0.0, σ_noise = 1.0)
    lag = GPDecompositions._LagIndex(t)
    v = GPDecompositions._lag_covariance(θ, lag.lags, GPDecompositions._default_kernels)
    C = cholesky(Symmetric(v[lag.index] + v[end] * I))
    w = GPDecompositions._innovation_whiteness(C, (y .- mean(y)) ./ std(y), t, Inf)
    @test w.pvalue < 0.01
    @test w.max_lag == 1 && w.max_acf > 0
end

@testset "lag-wise likelihood matches AbstractGPs" begin
    G = GPDecompositions
    Random.seed!(6)
    t = sort([60 .* rand(60); fill(20.0, 4); fill(41.5, 3)])    # irregular, replicates
    y = randn(length(t))
    kernels = G._resolve_kernels((trend = SEKernel(),
        seasonal = (effect = Matern32Kernel(), envelope = Matern52Kernel())))
    θ = (σ_trend = 1.1, l_trend = 9.0, σ_seasonal = 0.4, l_seasonal = 0.8, period = 7.0,
        ρ = 0.3, σ_noise = 0.2)
    lag = G._LagIndex(t)
    k = G._signal_kernel(G._component_kernels(θ, kernels))
    v₀ = G._lag_covariance(θ, lag.lags, kernels)
    @test v₀[lag.index] ≈ kernelmatrix(k, t)    # noise is independent across replicates
    @test v₀[end] ≈ θ.σ_noise^2
    @test G._nll(θ, lag, y, kernels) ≈ -logpdf(GP(k)(t, θ.σ_noise^2 + G._JITTER), y)
    # exact gradient against central differences in the unconstrained parameters
    x, unflatten, _ = G._flatten(G._init_params(θ, Symbol[]))
    v(x) = G._lag_covariance(unflatten(x), lag.lags, kernels)
    nll(x) = first(G._nll_lags(v(x), lag, y, false))
    _, w = G._nll_lags(v(x), lag, y, true)
    g = G.ForwardDiff.jacobian(v, x)' * w
    h = 1e-6
    fd = [(nll(x + h * e) - nll(x - h * e)) / 2h for e in eachcol(Matrix(I, 6, 6))]
    @test g ≈ fd rtol = 1e-5
end

@testset "kernels are stationary, scaled to unit variance" begin
    @test_throws ArgumentError GaussianProcess(; kernel = LinearKernel())
    @test_throws ArgumentError GaussianProcess(; kernel = PeriodicKernel())
    Random.seed!(9)
    t = collect(1.0:60.0)
    y = sin.(2π .* t ./ 30) .+ 0.3 .* sin.(2π .* t ./ 6) .+ 0.1 .* randn(60)
    unit = fit(GaussianProcess, y, t; period = 6.0, kernel = SEKernel())
    scaled = fit(GaussianProcess, y, t; period = 6.0, kernel = 3.0 * SEKernel())
    @test collect(values(scaled.θ)) ≈ collect(values(unit.θ)) rtol = 1e-4
end

@testset "replicated observations have independent noise" begin
    Random.seed!(10)
    t = repeat(1.0:40.0; inner = 2)
    y = sin.(2π .* t ./ 20) .+ 0.3 .* sin.(2π .* t ./ 5) .+ 0.2 .* randn(80)
    d = fit(GaussianProcess, y, t; period = 5.0)
    @test 0.1 < d.θ.σ_noise < 0.4
    @test d.noise.mean[1:2:end] ≉ d.noise.mean[2:2:end]
    @test d.trend.mean[1:2:end] ≈ d.trend.mean[2:2:end]
    @test d.trend.mean .+ d.seasonal.mean .+ d.noise.mean ≈ y rtol = 1e-4
    @test all(0 .< d.noise.std .< d.θ.σ_noise)
    # a new observation's noise is the prior
    p = predict(d, [10.5, 20.0])
    @test p.noise.mean == [0.0, 0.0]
    @test p.noise.std ≈ fill(d.θ.σ_noise, 2)
end

@testset "warm start from a previous fit" begin
    Random.seed!(8)
    t = collect(1.0:120.0)
    y = sin.(2π .* t ./ 60) .+ 0.4 .* sin.(2π .* t ./ 7) .+ 0.1 .* randn(120)
    d = fit(GaussianProcess, y, t; period = 7.0)
    warm = fit(GaussianProcess, y, t; period = 7.0, θ_init = d.θ)
    @test warm._fit_info.fit.loglik ≥ d._fit_info.fit.loglik - 1e-3
    @test warm._fit_info.fit.iterations < d._fit_info.fit.iterations
end

@testset "rate matches the closed-form derivative posterior" begin
    # SE trend: cov(f(s), f'(t)) = k(s, t)(s - t) / l², var f'(t) = σ² / l²
    Random.seed!(1)
    t = collect(1.0:80.0)
    y = 0.05t .+ sin.(t ./ 7) .+ 0.2randn(80)
    d = fit(GaussianProcess, y, t; kernel = SEKernel(),
        θ_fixed = (σ_trend = 1.0, l_trend = 15.0, σ_noise = 0.2))
    tp = collect(0.0:0.25:90.0)
    r = predict(d, tp).rate
    ȳ, s = mean(y), std(y)
    σ, l, σn = 1 / s, 15.0, 0.2 / s
    k(a, b) = σ^2 * exp(-(a - b)^2 / (2l^2))
    K = [k(a, b) for a in t, b in t] + (σn^2 + 1e-6) * I
    Dk = [k(a, b) * (a - b) / l^2 for a in t, b in tp]
    μ = s .* (Dk' * (K \ ((y .- ȳ) ./ s)))
    sd = s .* sqrt.(σ^2 / l^2 .- vec(sum(Dk .* (K \ Dk); dims = 1)))
    @test r.mean ≈ μ rtol = 1e-6
    @test r.std ≈ sd rtol = 1e-6
end

@testset "components sum to the observation posterior" begin
    Random.seed!(2)
    t = collect(1.0:48.0)
    y = sin.(2π .* t ./ 12) .+ 0.05t .+ 0.1 .* randn(48)
    d = fit(GaussianProcess, y, t; period = 12.0, θ_fixed = (bandwidth = 30.0,))
    total = d.trend.mean .+ d.seasonal.mean .+ d.noise.mean
    @test total ≈ y rtol = 1e-4
end

@testset "input validation" begin
    t = collect(1.0:30.0)
    @test_throws ArgumentError fit(GaussianProcess, fill(2.0, 30), t)
    @test_throws ArgumentError fit(GaussianProcess, [NaN; randn(29)], t)
    @test_throws DimensionMismatch fit(GaussianProcess, randn(29), t)
    # repeated times: the sampling interval for the whiteness lags ignores ties
    t2 = repeat(1.0:30.0; inner = 3)
    y2 = sin.(2π .* t2 ./ 6) .+ 0.1 .* randn(90)
    d = fit(GaussianProcess, y2, t2; period = 6.0, θ_fixed = (bandwidth = Inf,))
    @test d._fit_info.fit.whiteness.lags == 12
end
