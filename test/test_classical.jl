using GPDecompositions
using Logging
using Test
using Random

Random.seed!(1)

const _tc = collect(2000 .+ (0:119) ./ 12)    # monthly, 10 years
const _yc = sin.(2π .* _tc) .+ 0.2 .* (_tc .- 2000) .+ 0.1 .* randn(120)

@testset "$D basic API" for D in (X11Decomp, STLDecomp)
    d = fit(D, _yc, _tc; period=1.0)
    @test d isa D
    @test d isa AbstractDecomp
    @test d.t == _tc
    for c in (d.trend, d.seasonal, d.noise, d.rate)
        @test c isa Component
        @test length(c.mean) == 120
        @test all(isnan, c.std)
        @test isempty(c.lower) && isempty(c.upper) && isempty(c.sig)
    end
    @test d.trend.mean .+ d.seasonal.mean .+ d.noise.mean ≈ _yc
    @test (show(IOBuffer(), MIME"text/plain"(), d); true)
end

@testset "$D default t and unsorted input" for D in (X11Decomp, STLDecomp)
    d = fit(D, _yc; period=12)
    @test d.t == 1:120
    perm = reverse(eachindex(_tc))
    @test fit(D, _yc[perm], _tc[perm]; period=1.0).trend.mean ≈
        fit(D, _yc, _tc; period=1.0).trend.mean
end

@testset "$D predict (in-sample only)" for D in (X11Decomp, STLDecomp)
    d = fit(D, _yc, _tc; period=1.0)
    @test predict(d).trend.mean == d.trend.mean
    d2 = predict(d, _tc[(end - 11):end])
    @test d2 isa D
    @test d2.t == _tc[(end - 11):end]
    @test d2.seasonal.mean == d.seasonal.mean[(end - 11):end]
    @test_throws ArgumentError predict(d, [_tc[end] + 1/12])
    @test_throws ArgumentError predict(d, [_tc[1] + 1/24])
end

@testset "$D input validation" for D in (X11Decomp, STLDecomp)
    nonuniform = [_tc[1:(end - 1)]; _tc[end] + 1]
    @test_throws ArgumentError fit(D, _yc, nonuniform; period=1.0)
    @test_throws ArgumentError fit(D, _yc, _tc; period=0.1)    # non-integer p
    @test_throws DomainError fit(D, _yc, _tc; period=-1.0)
    @test_throws ArgumentError fit(D, _yc[1:20], _tc[1:20]; period=1.0)
    @test_throws DimensionMismatch fit(D, _yc, _tc[1:(end - 1)]; period=1.0)
end

@testset "X11Decomp options" begin
    d = fit(X11Decomp, _yc, _tc; period=1.0)
    f = d.θ.smoothers.trend
    @test f isa FittedSmoother{<:LocalPolynomial{HendersonKernel}}
    @test f.model.window == 13
    @test f.model.endpoints isa MinimumRevision{Float64}
    @test f.model.endpoints.degree == 1
    @test d.θ.estimated.trend_window == 13
    @test d.θ.estimated.ratio == f.model.endpoints.ratio
    @test haskey(d.θ.estimated, :msr)
    @test d.θ.smoothers.seasonal.model isa Union{SeasonalMovingAverage,StableSeasonal}
    m = fit(
        X11Decomp,
        _yc,
        _tc;
        period=1.0,
        trend=Henderson(; window=13, endpoints=musgrave(3.5)),
    )
    @test m.θ.smoothers.trend.model.endpoints.ratio ≈ 4 / (π * 3.5^2)
    @test m.trend.mean != d.trend.mean
    h23 = fit(X11Decomp, _yc, _tc; period=1.0, trend=Henderson(; window=23))
    @test h23.θ.smoothers.trend.model.window == 23
    s = fit(X11Decomp, _yc, _tc; period=1.0, seasonal=StableSeasonal())
    @test s.θ.smoothers.seasonal.model isa StableSeasonal
    @test !haskey(s.θ.estimated, :msr)
    dh = fit(X11Decomp, _yc, _tc; period=1.0, trend=Henderson(; endpoints=NoEndpoints()))
    @test all(isfinite, dh.trend.mean[7:(end - 6)])
    @test all(isnan, dh.trend.mean[[1:6; (end - 5):end]])
    @test all(isfinite, dh.rate.mean[7:(end - 6)])
    @test all(isfinite, dh.seasonal.mean)    # NaN preliminary ends are handled by phase
    ma = fit(X11Decomp, _yc, _tc; period=1.0, trend=MovingAverage())
    @test ma.θ.estimated.trend_width ≈ 1.0
    @test all(isnan, ma.trend.mean[[1:6; (end - 5):end]])
    @test all(isfinite, ma.trend.mean[7:(end - 6)])
end

@testset "X11Decomp period-general defaults" begin
    for (p, window) in ((2, 5), (4, 5), (7, 9), (12, 13), (24, 25))
        y = randn(10p)
        @test fit(X11Decomp, y; period=p).θ.smoothers.trend.model.window == window
    end
    # Seasonal filters needing more cycles than available are shortened
    d = fit(X11Decomp, randn(6 * 12); period=12, seasonal=SeasonalMovingAverage(; n=3, m=9))
    @test d.θ.smoothers.seasonal.model == SeasonalMovingAverage(; n=3, m=3)
    @test d.θ.smoothers.preliminary_seasonal.model == SeasonalMovingAverage(; n=3, m=3)
    d = fit(X11Decomp, randn(4 * 12); period=12)
    @test d.θ.smoothers.seasonal.model isa StableSeasonal
    @test d.θ.smoothers.preliminary_seasonal.model isa StableSeasonal
    @test_throws ArgumentError fit(X11Decomp, randn(2 * 12 + 11); period=12)
end

@testset "moving seasonality ratio (X-11 D9A)" begin
    # Pure noise: sd(ΔI) / sd(ΔS) = (10/7) / (√2/7) = 10/√2 for the 7-term average
    msr = GPDecompositions._moving_seasonality_ratio(randn(400 * 12), 12)
    @test msr ≈ 10 / √2 rtol = 0.05
    p, N = 12, 240
    moving = [
        (1 + 0.5sin(2π * t / 5p)) * 3sin(2π * t / p) + 0.4(t ÷ p) * cos(2π * t / p) for
        t in 1:N
    ]
    d = fit(X11Decomp, moving .+ 0.02randn(N); period=p)
    @test d.θ.estimated.msr < 2.5
    @test d.θ.smoothers.seasonal.model == SeasonalMovingAverage(; n=3, m=3)
    fixed = [3sin(2π * t / p) for t in 1:N]
    d = fit(X11Decomp, fixed .+ randn(N); period=p)
    @test d.θ.estimated.msr > 6.5
    @test d.θ.smoothers.seasonal.model == SeasonalMovingAverage(; n=3, m=9)
    d = fit(X11Decomp, randn(6p); period=p)
    @test haskey(d.θ.estimated, :msr_note)
end

@testset "X11Decomp reproduces linear trend plus fixed seasonal ($trend)" for trend in (
    Henderson(), LOESS(), LocalPolynomial(; degree=2, kernel=TricubeKernel())
)
    for p in (2, 3, 4, 7, 12), N in (12p, 12p + 5, 3p + p ÷ 2 + 1)
        s = randn(p)
        s .-= sum(s) / p
        seasonal = [s[mod1(t, p)] for t in 1:N]
        linear = [2 + 0.3t for t in 1:N]
        d = fit(X11Decomp, linear + seasonal; period=p, trend)
        @test maximum(abs, d.trend.mean - linear) < 1e-10
        @test maximum(abs, d.seasonal.mean - seasonal) < 1e-10
        @test maximum(abs, d.noise.mean) < 1e-10
        @test maximum(abs, d.rate.mean .- 0.3) < 1e-10
    end
end

@testset "X11Decomp with a moving-average trend reproduces it in the interior" begin
    for p in (4, 7, 12)
        N = 12p
        s = randn(p)
        s .-= sum(s) / p
        seasonal = [s[mod1(t, p)] for t in 1:N]
        linear = [2 + 0.3t for t in 1:N]
        d = fit(X11Decomp, linear + seasonal; period=p, trend=MovingAverage())
        inner = findall(isfinite, d.trend.mean)
        @test length(inner) ≥ N - p
        @test maximum(abs, d.trend.mean[inner] - linear[inner]) < 1e-10
        @test maximum(abs, d.seasonal.mean - seasonal) < 1e-10
    end
end

@testset "X11Decomp reports a derived preliminary trend" begin
    d = fit(X11Decomp, _yc, _tc; period=1.0, preliminary_trend=MovingAverage())
    @test d.θ.estimated.preliminary_trend_width ≈ 1.0
    @test d.θ.smoothers.preliminary_trend.model.width ≈ 1.0
end

@testset "X11Decomp reports an estimated preliminary ratio" begin
    preliminary_trend = Henderson(; endpoints=MinimumRevision())
    d = fit(X11Decomp, _yc, _tc; period=1.0, preliminary_trend)
    @test d.θ.estimated.preliminary_ratio ==
        d.θ.smoothers.preliminary_trend.model.endpoints.ratio
    @test haskey(d.θ.estimated, :preliminary_msre)
end

@testset "STLDecomp reports its derived trend window" begin
    @test_logs (:info, r"STLDecomp") fit(STLDecomp, _yc, _tc; period=1.0)
    @test_logs min_level = Logging.Info fit(
        STLDecomp, _yc, _tc; period=1.0, trend=LOESS(; span=0.2, degree=1)
    )
end

@testset "X11Decomp reports estimated hyperparameters once" begin
    logs, d = Test.collect_test_logs(() -> fit(X11Decomp, _yc, _tc; period=1.0))
    @test length(logs) == 1
    @test occursin("X11Decomp", logs[1].message)
    @test logs[1].group === :estimated
    @test Dict(logs[1].kwargs) == Dict(pairs(d.θ.estimated))
    @test_logs min_level = Logging.Info fit(
        X11Decomp,
        _yc,
        _tc;
        period=1.0,
        trend=Henderson(; window=13, endpoints=musgrave(3.5)),
        seasonal=SeasonalMovingAverage(; n=3, m=5),
    )
end

@testset "X11Decomp with a loess trend" begin
    for (p, neighbours) in ((4, 7), (7, 11), (12, 15))
        d = fit(X11Decomp, randn(10p); period=p, trend=LOESS())
        @test d.θ.estimated.trend_neighbours == neighbours
        f = d.θ.smoothers.trend
        @test f isa FittedSmoother{LOESS}
        @test f.model.span ≈ neighbours / 10p
        @test f.model.degree == 2
        @test d.θ.smoothers.preliminary_trend.model == f.model
    end
    # Loess interior weights with minimum-revision ends: the tricube local polynomial
    hybrid = LocalPolynomial(; window=13, degree=2, kernel=TricubeKernel())
    d = fit(X11Decomp, _yc, _tc; period=1.0, trend=hybrid)
    @test d.θ.smoothers.trend.model.endpoints isa MinimumRevision{Float64}
end

@testset "STLDecomp default trend" begin
    d = fit(STLDecomp, _yc, _tc; period=1.0)
    q = d.θ.estimated.trend_neighbours
    @test isodd(q)
    @test q ≥ 1.5 * 12 / (1 - 1.5 / 7)
    @test d.θ.smoothers.trend.model.span ≈ q / 120
    @test d.θ.smoothers.trend.model.degree == 1
end
