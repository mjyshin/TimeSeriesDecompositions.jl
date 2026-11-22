using TimeSeriesDecompositions
using Logging
using Test
using Random
using Statistics: mean

Random.seed!(1)

const _tc = collect(2000 .+ (0:119) ./ 12)    # monthly, 10 years
const _yc = sin.(2π .* _tc) .+ 0.2 .* (_tc .- 2000) .+ 0.1 .* randn(120)

@testset "$D basic API" for D in (X11, STL)
    d = fit(D, _yc, _tc; period = 1.0)
    @test d isa Decomposition{D}
    @test d.model == D()
    @test d.t == _tc
    for c in (d.trend, d.seasonal, d.noise, d.rate)
        @test c isa Component
        @test length(c.mean) == 120
        @test all(isnan, c.std)
        @test isempty(c.lower) && isempty(c.upper) && isempty(c.sig)
    end
    @test d.trend.mean .+ d.seasonal.mean .+ d.noise.mean ≈ _yc
    @test sum(d[c].mean for c in (:trend, :seasonal, :noise)) ≈ _yc
    @test d[:smoother] === d.smoother
    # by name through the nested structs: components, fitted smoothers, smoothers and
    # end-weight schemes
    @test d[:trend][:mean] === d.trend.mean
    f = d[:smoother][:trend_final]
    @test f[:values] === f.values
    @test f[:model][:degree] == f.model.degree
    if D === X11
        @test f[:model][:endpoints][:ratio] == d.θ[:ratio_trend_final]
    end
    @test (show(IOBuffer(), MIME"text/plain"(), d); true)
    # θ holds numbers only; the smoothers and fit information are elsewhere
    @test all(v -> v isa Real, values(d.θ))
    @test d.period == 1.0
    @test !haskey(d.θ, :period)
    @test d.smoother.trend_final isa FittedSmoother
    @test keys(d.smoother) == filter(!=(:iterations), fieldnames(D))
end

@testset "$D default t and unsorted input" for D in (X11, STL)
    d = fit(D, _yc; period = 12)
    @test d.t == 1:120
    perm = reverse(eachindex(_tc))
    @test fit(D, _yc[perm], _tc[perm]; period = 1.0).trend.mean ≈
          fit(D, _yc, _tc; period = 1.0).trend.mean
end

@testset "$D predict ($trend)" for D in (X11, STL),
    trend in (Henderson(), MovingAverage(), LOESS())

    d = fit(D, _yc, _tc; period = 1.0, trend_final = trend)
    for c in (:trend, :seasonal, :noise, :rate)
        @test isequal(getfield(predict(d), c).mean, getfield(d, c).mean)
    end
    # On-grid times (unsorted, repeated, computed in floating point) copy the fit
    idx = [100, 3, 3, 57]
    t_grid = 2000 .+ (idx .- 1) ./ 12
    d2 = predict(d, t_grid)
    @test d2 isa Decomposition{D}
    @test d2.t == t_grid
    @test isequal(d2.seasonal.mean, d.seasonal.mean[idx])
    @test isequal(d2.trend.mean, d.trend.mean[idx])
    @test isequal(d2.rate.mean, d.rate.mean[idx])
    @test isequal(predict(d2, _tc).trend.mean, d.trend.mean)    # predicts from the fit
    # Off the grid: continuous, noise undefined, rate the derivative of the trend
    t_off = _tc[60] .+ [-1e-7, 1e-7]    # beyond the 1e-8Δt snap to the grid
    off = predict(d, t_off)
    @test off.trend.mean ≈ fill(d.trend.mean[60], 2) atol = 1e-6
    @test off.seasonal.mean ≈ fill(d.seasonal.mean[60], 2) atol = 1e-6
    @test all(isnan, off.noise.mean)
    @test all(isnan, off.trend.std)
    τ, ε = _tc[60] + 0.3 / 12, 1e-6
    ends = predict(d, [τ - ε, τ, τ + ε]).trend.mean
    @test predict(d, [τ]).rate.mean[1] ≈ (ends[3] - ends[1]) / 2ε rtol = 1e-4
    @test isempty(predict(d, Float64[]).t)
    @test isequal(
        predict(d, _tc[[1, end]] .+ [-1e-12, 1e-12]).rate.mean, d.rate.mean[[1, end]]
    )
    @test (@inferred predict(d, [_tc[60] + 0.3 / 12])) isa Decomposition{D}
    @test_throws DomainError predict(d, [_tc[1] - 1e-6])
    @test_throws DomainError predict(d, [_tc[end] + 1e-6])
end

@testset "$D predict off the grid for period $p" for D in (X11, STL),
    p in (2, 5)

    N = 10p
    x = 1:N
    y = 0.1x .+ cospi.(2 .* x ./ p .+ 0.2) .* (p > 2) .+ 0.05 .* randn(N)
    d = fit(D, y, x; period = p)
    k = N ÷ 2
    near = predict(d, k .+ [-1e-6, 1e-6])
    @test near.seasonal.mean ≈ fill(d.seasonal.mean[k], 2) atol = 1e-4
    @test near.trend.mean ≈ fill(d.trend.mean[k], 2) atol = 1e-4
    @test near.rate.mean ≈ fill(d.rate.mean[k], 2) atol = 1e-4
    @test all(isfinite, predict(d, range(1, N; length = 3N)).seasonal.mean)
end

@testset "$D predict: NaN where the trend is undefined" for D in (X11, STL)
    d = fit(D, _yc, _tc; period = 1.0, trend_final = MovingAverage())
    between = predict(d, _tc[1:2] .+ 1 / 24)
    @test all(isnan, between.trend.mean) && all(isnan, between.rate.mean)
    @test all(isfinite, between.seasonal.mean)
end

@testset "$D input validation" for D in (X11, STL)
    nonuniform = [_tc[1:(end - 1)]; _tc[end] + 1]
    @test_throws ArgumentError fit(D, _yc, nonuniform; period = 1.0)
    @test_throws ArgumentError fit(D, _yc, _tc; period = 0.1)    # non-integer p
    @test_throws DomainError fit(D, _yc, _tc; period = -1.0)
    @test_throws ArgumentError fit(D, _yc[1:20], _tc[1:20]; period = 1.0)
    @test_throws DimensionMismatch fit(D, _yc, _tc[1:(end - 1)]; period = 1.0)
    # the stage keywords, checked when the model is built
    @test_throws ArgumentError D(; level = LOESS())
    @test_throws ArgumentError D(; trend_first = LOESS())
    @test_throws ArgumentError D(; trend = StableSeasonal())
    @test_throws ArgumentError D(; seasonal_final = LOESS())
    @test_throws ArgumentError D(; trend = 1.0)
    @test_throws ArgumentError D(; trend = LOESS(), trend_final = LOESS())
end

@testset "stage keywords" begin
    l, s = LOESS(), StableSeasonal()
    @test X11() == X11(; TimeSeriesDecompositions._X11_DEFAULTS...)
    # a component keyword sets every stage of that component
    m = X11(; trend = l)
    @test (m.trend_init, m.trend_interm, m.trend_final) == (l, l, l)
    @test m.seasonal_init == X11().seasonal_init
    m = X11(; seasonal = s)
    @test (m.seasonal_init, m.seasonal_final) == (s, s)
    # a stage keyword sets one stage and keeps the others
    m = X11(; trend_final = l)
    @test (m.trend_init, m.trend_interm, m.trend_final) == (MovingAverage(), nothing, l)
    @test STL(; trend = l).trend_final == l
    @test STL(; seasonal = s).seasonal_final == s
    # the fit and the positional constructor
    d = fit(X11(; trend = l), _yc, _tc; period = 1.0)
    @test d.smoother.trend_init.model isa LOESS
    @test d.smoother.trend_interm.model isa LOESS
    @test d.smoother.trend_final.model isa LOESS
    @test keys(d.smoother) == fieldnames(X11)
    @test_throws ArgumentError X11(l, s, nothing, nothing, s)
    # shown by the settings that differ from the defaults
    @test sprint(show, X11()) == "X11()"
    @test sprint(show, STL(; seasonal_final = SeasonalMovingAverage(; n = 3, m = 3))) ==
          "STL(seasonal_final=SeasonalMovingAverage(n=3, m=3))"
end

@testset "X11 options" begin
    d = fit(X11, _yc, _tc; period = 1.0)
    f = d.smoother.trend_final
    @test f isa FittedSmoother{<:LocalPolynomial{HendersonKernel}}
    @test f.model.bandwidth ≥ 1.0 - 1e-8    # at least 12 months
    @test f.model.endpoints isa MinimumRevision{Float64}
    @test f.model.endpoints.degree == 1
    @test d.θ.bandwidth_trend_final == f.model.bandwidth
    @test :bandwidth_trend_final ∈ d._fit_info.estimated
    @test haskey(d._fit_info.criteria, :cv_score_trend_final)
    @test d.θ.ratio_trend_final == f.model.endpoints.ratio
    @test :ratio_trend_final ∈ d._fit_info.estimated
    @test haskey(d._fit_info.criteria, :msre_trend_final)
    @test haskey(d._fit_info.criteria, :msr)
    @test d.θ.ratio_trend_interm == 0.0
    @test :ratio_trend_interm ∉ d._fit_info.estimated
    @test d.smoother.seasonal_final.model isa Union{SeasonalMovingAverage, StableSeasonal}
    musgrave_trend = Henderson(; bandwidth = 1.0, endpoints = musgrave(3.5))
    m = fit(X11(; trend_final = musgrave_trend), _yc, _tc; period = 1.0)
    @test m.smoother.trend_final.model.endpoints.ratio ≈ 4 / (π * 3.5^2)
    @test m.θ.ratio_trend_final ≈ 4 / (π * 3.5^2)
    @test :ratio_trend_final ∉ m._fit_info.estimated
    @test m.trend.mean != d.trend.mean
    h23 = fit(X11, _yc, _tc; period = 1.0, trend_final = Henderson(; bandwidth = 22 / 12))
    @test h23.smoother.trend_final.model.bandwidth == 22 / 12
    @test :bandwidth_trend_final ∉ h23._fit_info.derived
    s = fit(X11, _yc, _tc; period = 1.0,
        seasonal_final = StableSeasonal())
    @test s.smoother.seasonal_final.model isa StableSeasonal
    @test !haskey(s._fit_info.criteria, :msr)
    @test !haskey(s.θ, :n_seasonal_final)
    dh = fit(
        X11,
        _yc,
        _tc;
        period = 1.0,
        trend_final = Henderson(; bandwidth = 1.0, endpoints = NoEndpoints())
    )
    @test all(isfinite, dh.trend.mean[7:(end - 6)])
    @test all(isnan, dh.trend.mean[[1:6; (end - 5):end]])
    @test all(isfinite, dh.rate.mean[7:(end - 6)])
    @test all(isfinite, dh.seasonal.mean)    # NaN intermediate ends are handled by phase
    ma = fit(
        X11, _yc, _tc; period = 1.0, trend_final = MovingAverage())
    @test ma.θ.bandwidth_trend_final ≈ 1.0
    @test all(isnan, ma.trend.mean[[1:6; (end - 5):end]])
    @test all(isfinite, ma.trend.mean[7:(end - 6)])
end

@testset "X11 period-general defaults" begin
    # candidate trend lengths start at the period, so the trend cannot absorb the seasonal
    for (p, bandwidth) in ((2, 4), (4, 4), (7, 8), (12, 12), (24, 24))
        N = 10p
        m, derived = TimeSeriesDecompositions._resolve_trend(Henderson(), p, 1.0, N)
        @test first(m.bandwidth) == bandwidth && last(m.bandwidth) == 2 * ((N - 1) ÷ 4)
        @test isempty(derived)
        @test fit(X11, randn(N); period = p).θ.bandwidth_trend_final ≥ bandwidth
    end
    # Seasonal filters needing more cycles than available are shortened
    s39 = SeasonalMovingAverage(; n = 3, m = 9)
    d = fit(X11, randn(6 * 12); period = 12, seasonal_final = s39)
    @test d.smoother.seasonal_final.model == SeasonalMovingAverage(; n = 3, m = 3)
    @test (d.θ.n_seasonal_final, d.θ.m_seasonal_final) == (3, 3)
    @test haskey(d._fit_info.criteria, :note_seasonal_final)
    @test d.smoother.seasonal_init.model == SeasonalMovingAverage(; n = 3, m = 3)
    d = fit(X11, randn(4 * 12); period = 12)
    @test d.smoother.seasonal_final.model isa StableSeasonal
    @test d.smoother.seasonal_init.model isa StableSeasonal
    @test_throws ArgumentError fit(X11, randn(2 * 12 + 11); period = 12)
end

@testset "moving seasonality ratio (X-11 D9A)" begin
    # Pure noise: sd(ΔI) / sd(ΔS) = (10/7) / (√2/7) = 10/√2 for the 7-term average
    msr = TimeSeriesDecompositions._moving_seasonality_ratio(randn(400 * 12), 12)
    @test msr ≈ 10 / √2 rtol = 0.05
    p, N = 12, 240
    moving = [(1 + 0.5sin(2π * t / 5p)) * 3sin(2π * t / p) + 0.4(t ÷ p) * cos(2π * t / p)
              for
              t in 1:N]
    d = fit(X11, moving .+ 0.02randn(N); period = p)
    @test d._fit_info.criteria.msr < 2.5
    @test d.smoother.seasonal_final.model == SeasonalMovingAverage(; n = 3, m = 3)
    @test (d.θ.n_seasonal_final, d.θ.m_seasonal_final) == (3, 3)
    @test [:n_seasonal_final, :m_seasonal_final] ⊆ d._fit_info.estimated
    fixed = [3sin(2π * t / p) for t in 1:N]
    d = fit(X11, fixed .+ randn(N); period = p)
    @test d._fit_info.criteria.msr > 6.5
    @test d.smoother.seasonal_final.model == SeasonalMovingAverage(; n = 3, m = 9)
    d = fit(X11, randn(6p); period = p)
    @test haskey(d._fit_info.criteria, :note_msr)
    @test [:n_seasonal_final, :m_seasonal_final] ⊆ d._fit_info.derived
end

@testset "X11 reproduces linear trend plus fixed seasonal ($trend)" for trend in (
    Henderson(), LOESS(), LocalPolynomial(; degree = 2, kernel = TricubeKernel())
)
    for p in (2, 3, 4, 7, 12), N in (12p, 12p + 5, 3p + p ÷ 2 + 1)

        s = randn(p)
        s .-= sum(s) / p
        seasonal = [s[mod1(t, p)] for t in 1:N]
        linear = [2 + 0.3t for t in 1:N]
        d = fit(X11, linear + seasonal; period = p, trend_final = trend)
        @test maximum(abs, d.trend.mean - linear) < 1e-10
        @test maximum(abs, d.seasonal.mean - seasonal) < 1e-10
        @test maximum(abs, d.noise.mean) < 1e-10
        @test maximum(abs, d.rate.mean .- 0.3) < 1e-10
        # Off the grid: the line, and the periodic interpolation of the pattern
        c = range(1, N - p; length = 97)
        off = predict(d, c)
        @test maximum(abs, off.trend.mean - (2 .+ 0.3c)) < 1e-9
        @test maximum(abs, off.rate.mean .- 0.3) < 1e-9
        @test maximum(abs, off.seasonal.mean - predict(d, c .+ p).seasonal.mean) < 1e-9
    end
end

@testset "X11 with a moving-average trend reproduces it in the interior" begin
    for p in (4, 7, 12)
        N = 12p
        s = randn(p)
        s .-= sum(s) / p
        seasonal = [s[mod1(t, p)] for t in 1:N]
        linear = [2 + 0.3t for t in 1:N]
        d = fit(X11, linear + seasonal; period = p, trend_final = MovingAverage())
        inner = findall(isfinite, d.trend.mean)
        @test length(inner) ≥ N - p
        @test maximum(abs, d.trend.mean[inner] - linear[inner]) < 1e-10
        @test maximum(abs, d.seasonal.mean - seasonal) < 1e-10
    end
end

@testset "X11 intermediate trend" begin
    d = fit(X11, _yc, _tc; period = 1.0, trend_interm = MovingAverage())
    @test d.θ.bandwidth_trend_interm ≈ 1.0
    @test :bandwidth_trend_interm ∈ d._fit_info.derived
    @test d.smoother.trend_interm.model.bandwidth ≈ 1.0

    interm = Henderson(; endpoints = MinimumRevision())
    d = fit(X11, _yc, _tc; period = 1.0, trend_interm = interm)
    @test d.θ.ratio_trend_interm == d.smoother.trend_interm.model.endpoints.ratio
    @test :ratio_trend_interm ∈ d._fit_info.estimated
    @test haskey(d._fit_info.criteria, :msre_trend_interm)
end

@testset "STL reports what it chose from the data" begin
    @test_logs (:info, r"STL") fit(STL, _yc, _tc; period = 1.0)
    d = @quietly fit(STL, _yc, _tc; period = 1.0)
    @test d._fit_info.estimated ==
          [:bandwidth_seasonal_final, :span_trend_final, :neighbors_trend_final]
    @test isempty(d._fit_info.derived)
    @test first(keys(d._fit_info.criteria)) == :msr
    @test_logs (:info, r"STL") fit(
        STL, _yc, _tc; period = 1.0, trend_final = Henderson(; bandwidth = 2.0))
end

@testset "X11 reports in the order of fitting" begin
    d = fit(
        X11,
        _yc,
        _tc;
        period = 1.0,
        trend_interm = LOESS(; degree = 1),
        trend_final = LOESS(),
        seasonal_init = SeasonalMovingAverage(; n = 3, m = 9)
    )
    @test collect(keys(d.θ)) == [
        :bandwidth_trend_init,
        :n_seasonal_init,
        :m_seasonal_init,
        :span_trend_interm,
        :neighbors_trend_interm,
        :n_seasonal_final,
        :m_seasonal_final,
        :span_trend_final,
        :neighbors_trend_final
    ]
    @test (d.θ.n_seasonal_init, d.θ.m_seasonal_init) == (3, 5)    # shortened
    @test collect(keys(d._fit_info.criteria)) == [
        :note_seasonal_init,
        :cv_score_trend_interm,
        :enp_trend_interm,
        :autocorrelation_z_trend_interm,
        :msr,
        :note_seasonal_final,
        :cv_score_trend_final,
        :enp_trend_final,
        :autocorrelation_z_trend_final
    ]
    s = sprint(show, MIME"text/plain"(), d)
    @test occursin(r"neighbors_trend_interm +\d+\n", s)
    labels = ("smoothers", "trend_init", "seasonal_final", "estimated", "fixed", "criteria")
    for label in labels
        @test occursin(label, s)
    end
    @test findfirst("trend_init", s)[1] < findfirst("seasonal_init", s)[1] <
          findfirst("trend_interm", s)[1] < findfirst("seasonal_final", s)[1] <
          findfirst("trend_final", s)[1]
end

@testset "custom initial trend" begin
    d = fit(X11, _yc, _tc; period = 1.0)
    @test d.smoother.trend_init.model isa MovingAverage
    @test d.θ.bandwidth_trend_init ≈ 1.0
    @test :bandwidth_trend_init ∈ d._fit_info.derived
    h = Henderson(; bandwidth = 2.0, endpoints = DirectAsymmetric())
    d = fit(X11, _yc, _tc; period = 1.0, trend_init = h)
    @test d.smoother.trend_init.model == h
    @test all(isfinite, d.trend.mean)
    d = fit(X11, _yc, _tc; period = 1.0, trend_init = LOESS())
    @test d.θ.neighbors_trend_init == 15
    @test_throws ArgumentError STL(; trend_init = MovingAverage())    # starts from zero
end

@testset "X11 reports estimated hyperparameters once" begin
    logs, d = Test.collect_test_logs(() -> fit(X11, _yc, _tc; period = 1.0))
    @test length(logs) == 1
    @test occursin("X11", logs[1].message)
    @test logs[1].group === :estimated
    chosen = Dict(k => d.θ[k] for k in d._fit_info.estimated)
    @test Dict(logs[1].kwargs) == merge(chosen, Dict(pairs(d._fit_info.criteria)))
    @test_logs min_level = Logging.Info fit(
        X11,
        _yc,
        _tc;
        period = 1.0,
        trend_final = Henderson(; bandwidth = 1.0, endpoints = musgrave(3.5)),
        seasonal_final = SeasonalMovingAverage(; n = 3, m = 5)
    )
end

@testset "X11 with a loess trend" begin
    # candidate spans start at the smallest odd number ≥ p + 3 (at least 7) of neighbors
    for (p, neighbors) in ((4, 7), (7, 11), (12, 15))
        N = 10p
        m, _ = TimeSeriesDecompositions._resolve_trend(LOESS(), p, 1.0, N)
        @test round(Int, first(m.span) * N) == neighbors && last(m.span) == 1
        d = @quietly fit(X11, randn(N); period = p, trend_final = LOESS())
        @test d.θ.neighbors_trend_final ≥ neighbors
        f = d.smoother.trend_final
        @test f isa FittedSmoother{LOESS}
        @test d.θ.span_trend_final == f.model.span
        @test f.model.degree == 2
        @test d.smoother.trend_interm.model isa LOESS
    end
    # Loess interior weights with minimum-revision ends: the tricube local polynomial
    hybrid = LocalPolynomial(; bandwidth = 1.0, degree = 2, kernel = TricubeKernel())
    d = fit(X11, _yc, _tc; period = 1.0, trend_final = hybrid)
    @test d.smoother.trend_final.model.endpoints isa MinimumRevision{Float64}
end

@testset "STL default trend" begin
    d = fit(STL, _yc, _tc; period = 1.0)
    q = d.θ.neighbors_trend_final
    @test q ≥ 1.5 * 12 / (1 - 1.5 / 7)
    @test d.smoother.trend_final.model.span ≈ q / 120
    @test d.smoother.trend_final.model.degree == 2
    @test d.smoother.seasonal_final.model isa SeasonalLOESS
    @test d._fit_info.criteria.msr > 6.5    # a fixed pattern in noise: S3×9
    @test d.θ.bandwidth_seasonal_final ≈ 14.0
    @test collect(keys(d.θ)) ==
          [:bandwidth_seasonal_final, :span_trend_final, :neighbors_trend_final]
end

@testset "STL reproduces a linear trend and a fixed seasonal pattern" begin
    s = repeat([3, 1, -2, 0, 2, -1, -4, 1, 0, 2, -1, -1.0], 10)
    t = 1:120
    for N in (120, 115)    # also a part cycle
        d = @quietly fit(STL(), 0.3 .* t[1:N] .+ s[1:N]; period = 12)
        @test d.trend.mean ≈ 0.3 .* t[1:N] atol = 1e-10
        @test d.seasonal.mean ≈ s[1:N] atol = 1e-10
    end
end

@testset "STL matches R's stl without robustness" begin
    for line in eachline(joinpath(@__DIR__, "data", "stl_r.txt"))
        row = parse.(Float64, split(line))
        p, N, ns, nt, inner = Int.(row[1:5])
        y, trend, seasonal = row[6:(5 + N)], row[(6 + N):(5 + 2N)], row[(6 + 2N):end]
        model = STL(; seasonal = SeasonalLOESS(; bandwidth = (ns - 1) * p),
            trend = LOESS(; span = nt / N, degree = 1), iterations = inner)
        d = fit(model, y; period = p)
        @test d.trend.mean ≈ trend rtol = 1e-10
        @test d.seasonal.mean ≈ seasonal rtol = 1e-10
    end
end

@testset "STL options" begin
    @test STL().iterations == 2
    @test sprint(show, STL(; iterations = 1)) == "STL(iterations=1)"
    @test_throws DomainError STL(; iterations = 0)
    one, two = [@quietly(fit(STL(; iterations = k), _yc, _tc; period = 1.0)) for k in 1:2]
    @test one.seasonal.mean != two.seasonal.mean
    # the seasonal smoother of X11, and X11's in STL, extended by the nearest same phase
    d = fit(
        STL(; seasonal = SeasonalMovingAverage(), trend = Henderson(; bandwidth = 1.0)),
        _yc, _tc; period = 1.0)
    @test (d.θ.n_seasonal_final, d.θ.m_seasonal_final) == (3, 5)
    @test d.trend.mean .+ d.seasonal.mean .+ d.noise.mean ≈ _yc
    d = @quietly fit(X11(; seasonal = SeasonalLOESS()), _yc, _tc; period = 1.0)
    @test d.θ.bandwidth_seasonal_init ≈ 6.0    # fixed, like S3×3
    @test :bandwidth_seasonal_init ∈ d._fit_info.derived
    @test d.θ.bandwidth_seasonal_final ≈ 14.0    # by the MSR, like S3×9
    @test :bandwidth_seasonal_final ∈ d._fit_info.estimated
end

@testset "SeasonalLOESS bandwidth by the MSR" begin
    # windows matching the variance reduction of S3×3, S3×5 and S3×9
    matched(n, m, degree) = TimeSeriesDecompositions._matched_loess(
        SeasonalMovingAverage(n, m), degree, 7.0).bandwidth
    @test [matched(3, m, 1) for m in (3, 5, 9)] ≈ 7 .* [6, 8, 14]
    @test [matched(3, m, 2) for m in (3, 5, 9)] ≈ 7 .* [12, 18, 28]
    # a given bandwidth is kept
    d = fit(STL(; seasonal = SeasonalLOESS(; bandwidth = 10.0)), _yc, _tc; period = 1.0)
    @test d.θ.bandwidth_seasonal_final ≈ 10.0
    @test !haskey(d._fit_info.criteria, :msr)
    # below seven cycles: six periods, with a note
    d = @quietly fit(STL, _yc[1:72], _tc[1:72]; period = 1.0)
    @test d.θ.bandwidth_seasonal_final ≈ 6.0
    @test :bandwidth_seasonal_final ∈ d._fit_info.derived
    @test occursin("six periods", d._fit_info.criteria.note_msr)
end

@testset "X11 trend length near the interior optimum" begin
    # smooth trend, weekly pattern and independent noise: leave-one-out targets the error of
    # the trend, and comes close to the best interior estimate of it
    rng = Xoshiro(7)
    N, p = 300, 7
    t = 1:N
    trend = 0.8 .* sin.(2π .* t ./ 90) .+ 0.3 .* sin.(2π .* t ./ 37)
    rate = 0.8 .* (2π / 90) .* cos.(2π .* t ./ 90) .+
           0.3 .* (2π / 37) .* cos.(2π .* t ./ 37)
    y = trend .+ 0.3 .* sin.(2π .* t ./ p) .+ 0.08 .* randn(rng, N)
    inner = 41:(N - 40)
    err(d, c, truth) = sqrt(mean(abs2, d[c].mean[inner] .- truth[inner]))
    d = @quietly fit(X11(), y; period = p)
    fits = [@quietly fit(X11(; trend_final = Henderson(; bandwidth = b)), y; period = p)
            for b in 8:2:44]
    @test err(d, :trend, trend) ≤ 1.1 * minimum(f -> err(f, :trend, trend), fits)
    # the shortest candidate is the old fixed default
    @test err(d, :rate, rate) < 0.5 * err(fits[1], :rate, rate)
end
