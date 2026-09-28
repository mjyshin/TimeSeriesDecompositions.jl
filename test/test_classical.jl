using GPDecompositions
using Logging
using Test
using Random

Random.seed!(1)

const _tc = collect(2000 .+ (0:119) ./ 12)    # monthly, 10 years
const _yc = sin.(2π .* _tc) .+ 0.2 .* (_tc .- 2000) .+ 0.1 .* randn(120)

final_trend(smoother) = (trend=(final=smoother,),)

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
    # θ holds numbers only; the smoothers and fit information are elsewhere
    @test all(v -> v isa Real, values(d.θ))
    @test d.θ.period == 1.0
    @test d.smoother.trend.final isa FittedSmoother
    @test d.smoother.trend.init.model isa MovingAverage
end

@testset "$D default t and unsorted input" for D in (X11Decomp, STLDecomp)
    d = fit(D, _yc; period=12)
    @test d.t == 1:120
    perm = reverse(eachindex(_tc))
    @test fit(D, _yc[perm], _tc[perm]; period=1.0).trend.mean ≈
        fit(D, _yc, _tc; period=1.0).trend.mean
end

@testset "$D predict ($trend)" for D in (X11Decomp, STLDecomp),
    trend in (Henderson(), MovingAverage(), LOESS())

    d = fit(D, _yc, _tc; period=1.0, smoother=final_trend(trend))
    for c in (:trend, :seasonal, :noise, :rate)
        @test isequal(getfield(predict(d), c).mean, getfield(d, c).mean)
    end
    # On-grid times (unsorted, repeated, computed in floating point) copy the fit
    idx = [100, 3, 3, 57]
    t_grid = 2000 .+ (idx .- 1) ./ 12
    d2 = predict(d, t_grid)
    @test d2 isa D
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
    @test (@inferred predict(d, [_tc[60] + 0.3 / 12])) isa D
    @test_throws DomainError predict(d, [_tc[1] - 1e-6])
    @test_throws DomainError predict(d, [_tc[end] + 1e-6])
end

@testset "$D predict off the grid for period $p" for D in (X11Decomp, STLDecomp),
    p in (2, 5)

    N = 10p
    x = 1:N
    y = 0.1x .+ cospi.(2 .* x ./ p .+ 0.2) .* (p > 2) .+ 0.05 .* randn(N)
    d = fit(D, y, x; period=p)
    k = N ÷ 2
    near = predict(d, k .+ [-1e-6, 1e-6])
    @test near.seasonal.mean ≈ fill(d.seasonal.mean[k], 2) atol = 1e-4
    @test near.trend.mean ≈ fill(d.trend.mean[k], 2) atol = 1e-4
    @test near.rate.mean ≈ fill(d.rate.mean[k], 2) atol = 1e-4
    @test all(isfinite, predict(d, range(1, N; length=3N)).seasonal.mean)
end

@testset "$D predict: NaN where the trend is undefined" for D in (X11Decomp, STLDecomp)
    d = fit(D, _yc, _tc; period=1.0, smoother=final_trend(MovingAverage()))
    between = predict(d, _tc[1:2] .+ 1 / 24)
    @test all(isnan, between.trend.mean) && all(isnan, between.rate.mean)
    @test all(isfinite, between.seasonal.mean)
end

@testset "$D input validation" for D in (X11Decomp, STLDecomp)
    nonuniform = [_tc[1:(end - 1)]; _tc[end] + 1]
    @test_throws ArgumentError fit(D, _yc, nonuniform; period=1.0)
    @test_throws ArgumentError fit(D, _yc, _tc; period=0.1)    # non-integer p
    @test_throws DomainError fit(D, _yc, _tc; period=-1.0)
    @test_throws ArgumentError fit(D, _yc[1:20], _tc[1:20]; period=1.0)
    @test_throws DimensionMismatch fit(D, _yc, _tc[1:(end - 1)]; period=1.0)
    # the smoother keyword
    @test_throws ArgumentError fit(D, _yc, _tc; period=1.0, smoother=(level=LOESS(),))
    @test_throws ArgumentError fit(
        D, _yc, _tc; period=1.0, smoother=(trend=(first=LOESS(),),)
    )
    @test_throws ArgumentError fit(D, _yc, _tc; period=1.0, smoother=(trend=StableSeasonal(),))
    @test_throws ArgumentError fit(
        D, _yc, _tc; period=1.0, smoother=(seasonal=(final=LOESS(),),)
    )
    @test_throws ArgumentError fit(D, _yc, _tc; period=1.0, smoother=(trend=1.0,))
end

@testset "smoother keyword" begin
    R = GPDecompositions._resolve_smoothers
    D = GPDecompositions._X11_SMOOTHERS
    @test R((;), D) == D
    # one smoother applies to every stage of its component
    l = LOESS()
    @test R((trend=l,), D).trend == (init=l, interm=l, final=l)
    @test R((trend=l,), D).seasonal == D.seasonal
    s = StableSeasonal()
    @test R((seasonal=s,), D).seasonal == (init=s, final=s)
    # a NamedTuple sets some stages and keeps the others
    @test R((trend=(final=l,),), D).trend == (init=MovingAverage(), interm=nothing, final=l)
    @test R((seasonal=(final=s,),), D).seasonal.init == SeasonalMovingAverage(; n=3, m=3)
    d = fit(X11Decomp, _yc, _tc; period=1.0, smoother=(trend=LOESS(),))
    @test d.smoother.trend.init.model isa LOESS
    @test d.smoother.trend.interm.model isa LOESS
    @test d.smoother.trend.final.model isa LOESS
end

@testset "X11Decomp options" begin
    d = fit(X11Decomp, _yc, _tc; period=1.0)
    f = d.smoother.trend.final
    @test f isa FittedSmoother{<:LocalPolynomial{HendersonKernel}}
    @test f.model.bandwidth ≈ 1.0    # 12 months
    @test f.model.endpoints isa MinimumRevision{Float64}
    @test f.model.endpoints.degree == 1
    @test d.θ.bandwidth_trend_final ≈ 1.0
    @test :bandwidth_trend_final ∈ d._fit_info.derived
    @test d.θ.ratio_trend_final == f.model.endpoints.ratio
    @test :ratio_trend_final ∈ d._fit_info.estimated
    @test haskey(d._fit_info.criteria, :msre_trend_final)
    @test haskey(d._fit_info.criteria, :msr)
    @test d.θ.ratio_trend_interm == 0.0
    @test :ratio_trend_interm ∉ d._fit_info.estimated
    @test d.smoother.seasonal.final.model isa Union{SeasonalMovingAverage,StableSeasonal}
    musgrave_trend = Henderson(; bandwidth=1.0, endpoints=musgrave(3.5))
    m = fit(X11Decomp, _yc, _tc; period=1.0, smoother=final_trend(musgrave_trend))
    @test m.smoother.trend.final.model.endpoints.ratio ≈ 4 / (π * 3.5^2)
    @test m.θ.ratio_trend_final ≈ 4 / (π * 3.5^2)
    @test :ratio_trend_final ∉ m._fit_info.estimated
    @test m.trend.mean != d.trend.mean
    h23 = fit(
        X11Decomp, _yc, _tc; period=1.0, smoother=final_trend(Henderson(; bandwidth=22 / 12))
    )
    @test h23.smoother.trend.final.model.bandwidth == 22 / 12
    @test :bandwidth_trend_final ∉ h23._fit_info.derived
    s = fit(X11Decomp, _yc, _tc; period=1.0, smoother=(seasonal=(final=StableSeasonal(),),))
    @test s.smoother.seasonal.final.model isa StableSeasonal
    @test !haskey(s._fit_info.criteria, :msr)
    @test !haskey(s.θ, :n_seasonal_final)
    dh = fit(
        X11Decomp,
        _yc,
        _tc;
        period=1.0,
        smoother=final_trend(Henderson(; endpoints=NoEndpoints())),
    )
    @test all(isfinite, dh.trend.mean[7:(end - 6)])
    @test all(isnan, dh.trend.mean[[1:6; (end - 5):end]])
    @test all(isfinite, dh.rate.mean[7:(end - 6)])
    @test all(isfinite, dh.seasonal.mean)    # NaN intermediate ends are handled by phase
    ma = fit(X11Decomp, _yc, _tc; period=1.0, smoother=final_trend(MovingAverage()))
    @test ma.θ.bandwidth_trend_final ≈ 1.0
    @test all(isnan, ma.trend.mean[[1:6; (end - 5):end]])
    @test all(isfinite, ma.trend.mean[7:(end - 6)])
end

@testset "X11Decomp period-general defaults" begin
    for (p, bandwidth) in ((2, 4), (4, 4), (7, 8), (12, 12), (24, 24))
        y = randn(10p)
        @test fit(X11Decomp, y; period=p).smoother.trend.final.model.bandwidth == bandwidth
    end
    # Seasonal filters needing more cycles than available are shortened
    s39 = (seasonal=(final=SeasonalMovingAverage(; n=3, m=9),),)
    d = fit(X11Decomp, randn(6 * 12); period=12, smoother=s39)
    @test d.smoother.seasonal.final.model == SeasonalMovingAverage(; n=3, m=3)
    @test (d.θ.n_seasonal_final, d.θ.m_seasonal_final) == (3, 3)
    @test haskey(d._fit_info.criteria, :note_seasonal_final)
    @test d.smoother.seasonal.init.model == SeasonalMovingAverage(; n=3, m=3)
    d = fit(X11Decomp, randn(4 * 12); period=12)
    @test d.smoother.seasonal.final.model isa StableSeasonal
    @test d.smoother.seasonal.init.model isa StableSeasonal
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
    @test d._fit_info.criteria.msr < 2.5
    @test d.smoother.seasonal.final.model == SeasonalMovingAverage(; n=3, m=3)
    @test (d.θ.n_seasonal_final, d.θ.m_seasonal_final) == (3, 3)
    @test [:n_seasonal_final, :m_seasonal_final] ⊆ d._fit_info.estimated
    fixed = [3sin(2π * t / p) for t in 1:N]
    d = fit(X11Decomp, fixed .+ randn(N); period=p)
    @test d._fit_info.criteria.msr > 6.5
    @test d.smoother.seasonal.final.model == SeasonalMovingAverage(; n=3, m=9)
    d = fit(X11Decomp, randn(6p); period=p)
    @test haskey(d._fit_info.criteria, :note_msr)
    @test [:n_seasonal_final, :m_seasonal_final] ⊆ d._fit_info.derived
end

@testset "X11Decomp reproduces linear trend plus fixed seasonal ($trend)" for trend in (
    Henderson(), LOESS(), LocalPolynomial(; degree=2, kernel=TricubeKernel())
)
    for p in (2, 3, 4, 7, 12), N in (12p, 12p + 5, 3p + p ÷ 2 + 1)
        s = randn(p)
        s .-= sum(s) / p
        seasonal = [s[mod1(t, p)] for t in 1:N]
        linear = [2 + 0.3t for t in 1:N]
        d = fit(X11Decomp, linear + seasonal; period=p, smoother=final_trend(trend))
        @test maximum(abs, d.trend.mean - linear) < 1e-10
        @test maximum(abs, d.seasonal.mean - seasonal) < 1e-10
        @test maximum(abs, d.noise.mean) < 1e-10
        @test maximum(abs, d.rate.mean .- 0.3) < 1e-10
        # Off the grid: the line, and the periodic interpolation of the pattern
        c = range(1, N - p; length=97)
        off = predict(d, c)
        @test maximum(abs, off.trend.mean - (2 .+ 0.3c)) < 1e-9
        @test maximum(abs, off.rate.mean .- 0.3) < 1e-9
        @test maximum(abs, off.seasonal.mean - predict(d, c .+ p).seasonal.mean) < 1e-9
    end
end

@testset "X11Decomp with a moving-average trend reproduces it in the interior" begin
    for p in (4, 7, 12)
        N = 12p
        s = randn(p)
        s .-= sum(s) / p
        seasonal = [s[mod1(t, p)] for t in 1:N]
        linear = [2 + 0.3t for t in 1:N]
        d = fit(X11Decomp, linear + seasonal; period=p, smoother=final_trend(MovingAverage()))
        inner = findall(isfinite, d.trend.mean)
        @test length(inner) ≥ N - p
        @test maximum(abs, d.trend.mean[inner] - linear[inner]) < 1e-10
        @test maximum(abs, d.seasonal.mean - seasonal) < 1e-10
    end
end

@testset "X11Decomp intermediate trend" begin
    d = fit(X11Decomp, _yc, _tc; period=1.0, smoother=(trend=(interm=MovingAverage(),),))
    @test d.θ.bandwidth_trend_interm ≈ 1.0
    @test :bandwidth_trend_interm ∈ d._fit_info.derived
    @test d.smoother.trend.interm.model.bandwidth ≈ 1.0

    interm = Henderson(; endpoints=MinimumRevision())
    d = fit(X11Decomp, _yc, _tc; period=1.0, smoother=(trend=(; interm),))
    @test d.θ.ratio_trend_interm == d.smoother.trend.interm.model.endpoints.ratio
    @test :ratio_trend_interm ∈ d._fit_info.estimated
    @test haskey(d._fit_info.criteria, :msre_trend_interm)
end

@testset "STLDecomp reports what it chose from the data" begin
    @test_logs min_level = Logging.Info fit(STLDecomp, _yc, _tc; period=1.0)
    d = fit(STLDecomp, _yc, _tc; period=1.0)
    @test [:span_trend_final, :neighbors_trend_final] ⊆ d._fit_info.derived
    @test isempty(d._fit_info.estimated)
    henderson = final_trend(Henderson(; bandwidth=2.0))
    @test_logs (:info, r"STLDecomp") fit(STLDecomp, _yc, _tc; period=1.0, smoother=henderson)
end

@testset "X11Decomp reports in the order of fitting" begin
    d = fit(
        X11Decomp,
        _yc,
        _tc;
        period=1.0,
        smoother=(
            trend=(interm=LOESS(; degree=1), final=LOESS()),
            seasonal=(init=SeasonalMovingAverage(; n=3, m=9),),
        ),
    )
    @test collect(keys(d.θ)) == [
        :period,
        :bandwidth_trend_init,
        :n_seasonal_init,
        :m_seasonal_init,
        :span_trend_interm,
        :neighbors_trend_interm,
        :n_seasonal_final,
        :m_seasonal_final,
        :span_trend_final,
        :neighbors_trend_final,
    ]
    @test (d.θ.n_seasonal_init, d.θ.m_seasonal_init) == (3, 5)    # shortened
    @test collect(keys(d._fit_info.criteria)) ==
        [:note_seasonal_init, :msr, :note_seasonal_final]
    s = sprint(show, MIME"text/plain"(), d)
    @test occursin(r"neighbors_trend_interm +15 \(derived\)", s)
    for label in ("smoothers", "trend.init", "seasonal.final", "estimated", "fixed", "criteria")
        @test occursin(label, s)
    end
    @test findfirst("trend.init", s)[1] < findfirst("seasonal.init", s)[1] <
        findfirst("trend.interm", s)[1] < findfirst("seasonal.final", s)[1] <
        findfirst("trend.final", s)[1]
end

@testset "custom initial trend" begin
    d = fit(X11Decomp, _yc, _tc; period=1.0)
    @test d.smoother.trend.init.model isa MovingAverage
    @test d.θ.bandwidth_trend_init ≈ 1.0
    @test :bandwidth_trend_init ∈ d._fit_info.derived
    h = Henderson(; bandwidth=2.0, endpoints=DirectAsymmetric())
    d = fit(X11Decomp, _yc, _tc; period=1.0, smoother=(trend=(init=h,),))
    @test d.smoother.trend.init.model == h
    @test all(isfinite, d.trend.mean)
    d = fit(X11Decomp, _yc, _tc; period=1.0, smoother=(trend=(init=LOESS(),),))
    @test d.θ.neighbors_trend_init == 15
    s = fit(
        STLDecomp, _yc, _tc; period=1.0, smoother=(trend=(init=Henderson(; bandwidth=2.0),),)
    )
    @test s.smoother.trend.init.model.bandwidth == 2.0
    @test first(s._fit_info.estimated) == :ratio_trend_init
end

@testset "X11Decomp reports estimated hyperparameters once" begin
    logs, d = Test.collect_test_logs(() -> fit(X11Decomp, _yc, _tc; period=1.0))
    @test length(logs) == 1
    @test occursin("X11Decomp", logs[1].message)
    @test logs[1].group === :estimated
    chosen = Dict(k => d.θ[k] for k in d._fit_info.estimated)
    @test Dict(logs[1].kwargs) == merge(chosen, Dict(pairs(d._fit_info.criteria)))
    @test_logs min_level = Logging.Info fit(
        X11Decomp,
        _yc,
        _tc;
        period=1.0,
        smoother=(
            trend=(final=Henderson(; bandwidth=1.0, endpoints=musgrave(3.5)),),
            seasonal=(final=SeasonalMovingAverage(; n=3, m=5),),
        ),
    )
end

@testset "X11Decomp with a loess trend" begin
    for (p, neighbors) in ((4, 7), (7, 11), (12, 15))
        d = fit(X11Decomp, randn(10p); period=p, smoother=final_trend(LOESS()))
        @test d.θ.neighbors_trend_final == neighbors
        f = d.smoother.trend.final
        @test f isa FittedSmoother{LOESS}
        @test f.model.span ≈ neighbors / 10p
        @test d.θ.span_trend_final ≈ neighbors / 10p
        @test f.model.degree == 2
        @test d.smoother.trend.interm.model == f.model
    end
    # Loess interior weights with minimum-revision ends: the tricube local polynomial
    hybrid = LocalPolynomial(; bandwidth=1.0, degree=2, kernel=TricubeKernel())
    d = fit(X11Decomp, _yc, _tc; period=1.0, smoother=final_trend(hybrid))
    @test d.smoother.trend.final.model.endpoints isa MinimumRevision{Float64}
end

@testset "STLDecomp default trend" begin
    d = fit(STLDecomp, _yc, _tc; period=1.0)
    q = d.θ.neighbors_trend_final
    @test isodd(q)
    @test q ≥ 1.5 * 12 / (1 - 1.5 / 7)
    @test d.smoother.trend.final.model.span ≈ q / 120
    @test d.smoother.trend.final.model.degree == 1
    @test d.smoother.seasonal.final.model isa StableSeasonal
end
