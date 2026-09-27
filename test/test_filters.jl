using GPDecompositions
using LinearAlgebra: dot
using Logging
using Random
using Test

Random.seed!(2)
const _y = [sin(2π * t / 12) + 0.02t + 0.3randn() for t in 1:120]

# Henderson weights in closed form (X-11; m = h + 2)
function henderson_closed_form(h)
    m = h + 2
    return map((-h):h) do j
        num =
            315 * ((m - 1)^2 - j^2) * (m^2 - j^2) * ((m + 1)^2 - j^2) * (3m^2 - 16 - 11j^2)
        return num / (8m * (m^2 - 1) * (4m^2 - 1) * (4m^2 - 9) * (4m^2 - 25))
    end
end

# Musgrave end weights in closed form (Proietti & Luati, 2008, eq. 11)
function musgrave_closed_form(w, M, R)
    H = length(w)
    β = 4 / (π * R^2)
    c = (M + 1) / 2
    tail = sum(w[(M + 1):H])
    slope = sum((j - c) * w[j] for j in (M + 1):H) / (1 + β * M * (M + 1) * (M - 1) / 12)
    return [w[i] + tail / M + β * (i - c) * slope for i in 1:M]
end

lpf(; kwargs...) = fit(LocalPolynomialFilter, _y; kwargs...)

@testset "kernel_weights" begin
    @test kernel_weights(HendersonKernel(), 6) ==
        [(49 - j^2) * (64 - j^2) * (81 - j^2) for j in -6:6]
    for k in (
        HendersonKernel(),
        BiweightKernel(),
        TricubeKernel(),
        EpanechnikovKernel(),
        UniformKernel(),
    )
        κ = kernel_weights(k, 4)
        @test κ == reverse(κ)
        @test all(>(0), κ)
        @test lpf(; kernel=k, endpoints=NoEndpoints()).weights |> sum ≈ 1
    end
end

@testset "Henderson symmetric weights" begin
    # Published 13-term weights, j = 0:6
    f = lpf(; window=13, endpoints=NoEndpoints())
    @test f.weights[7:13] ≈
        [0.240057, 0.214337, 0.147357, 0.065492, 0.0, -0.027864, -0.019350] atol = 1e-6
    for window in (5, 7, 9, 13, 23, 51, 101)
        h = (window - 1) ÷ 2
        y = randn(max(window, 120))
        w = fit(LocalPolynomialFilter, y; window, endpoints=NoEndpoints()).weights
        @test w ≈ henderson_closed_form(h) atol = 1e-10
        @test sum(w) ≈ 1
    end
    @test lpf(; window=9, endpoints=NoEndpoints()).weights[5:9] ≈
        [0.331139, 0.266557, 0.118470, -0.009872, -0.040724] atol = 1e-6
end

@testset "direct asymmetric weights (P&L Table 1)" begin
    # Weight on the current observation, h = 6; rows q = 0:6, columns d = 0:6
    table = [
        0.245665 0.585646 0.835625 0.955184 0.992471 0.999412 1.000000
        0.199086 0.303839 0.306032 0.455995 0.728452 0.923773 0.990781
        0.171191 0.200760 0.265311 0.427533 0.449289 0.518942 0.766170
        0.154679 0.161543 0.265235 0.338531 0.360310 0.514373 0.539702
        0.145554 0.146624 0.257785 0.277607 0.357651 0.430913 0.459352
        0.141309 0.141375 0.247161 0.249482 0.351565 0.364447 0.459300
        0.140033 0.140033 0.240057 0.240057 0.337907 0.337907 0.441782
    ]
    for d in 0:6
        f = lpf(; window=13, degree=d, endpoints=DirectAsymmetric())
        for q in 0:5
            @test f.end_weights[q + 1][7] ≈ table[q + 1, d + 1] atol = 1e-6
        end
        @test f.weights[7] ≈ table[7, d + 1] atol = 1e-6
    end
    @test lpf(; endpoints=DirectAsymmetric()).end_weights[1] ≈
        [-0.017237, 0.021887, 0.040002, -0.034147, -0.097894, 0.132204, 0.955184] atol =
        1e-6
end

@testset "Musgrave end weights" begin
    f = lpf(; window=13, endpoints=musgrave(3.5))
    @test f.endpoints.ratio ≈ 0.103938 atol = 1e-6
    published = [
        [-0.091860, -0.058110, 0.012018, 0.119773, 0.243902, 0.353146, 0.421131],
        [-0.042707, -0.038632, 0.001821, 0.079902, 0.174355, 0.253925, 0.292234, 0.279102],
        [
            -0.016033,
            -0.024868,
            0.002674,
            0.067844,
            0.149387,
            0.216046,
            0.241445,
            0.215403,
            0.148101,
        ],
    ]
    for q in 0:2
        @test f.end_weights[q + 1] ≈ published[q + 1] atol = 1e-6
    end
    @test lpf(; window=9, endpoints=musgrave(1.0)).end_weights[1] ≈
        [-0.155536, -0.033836, 0.185356, 0.424292, 0.579724] atol = 1e-6
    for (window, R) in ((9, 1.0), (13, 3.5), (23, 4.5))
        f = lpf(; window, endpoints=musgrave(R))
        h = (window - 1) ÷ 2
        for q in 0:(h - 1)
            @test f.end_weights[q + 1] ≈ musgrave_closed_form(f.weights, h + 1 + q, R) atol =
                1e-12
        end
    end
    @test musgrave_ic_ratio(9) == 1.0
    @test musgrave_ic_ratio(13) == 3.5
    @test musgrave_ic_ratio(23) == 4.5
    @test_throws ArgumentError musgrave_ic_ratio(15)
end

@testset "minimum-revision weights" begin
    ql = lpf(; endpoints=MinimumRevision(; degree=1, ratio=0.016))
    @test ql.end_weights[1] ≈
        [0.021586, -0.087155, -0.096713, -0.005838, 0.164216, 0.382191, 0.621712] atol =
        1e-6
    cq = lpf(; endpoints=MinimumRevision(; degree=2, ratio=0.003))
    @test cq.end_weights[1] ≈
        [0.064348, -0.013092, -0.092745, -0.099298, 0.020058, 0.308128, 0.812601] atol =
        1e-6

    # Reproduction constraints Uₚᵀv = Uᵀw for polynomials of degree d*
    h, js = 6, -6:6
    for degree in 0:2, q in 0:(h - 1)
        f = lpf(; endpoints=MinimumRevision(; degree, ratio=0.05))
        v, p = f.end_weights[q + 1], 1:(h + 1 + q)
        for k in 0:degree
            @test dot(v, js[p] .^ k) ≈ dot(f.weights, js .^ k) atol = 1e-9
        end
    end

    # ratio → ∞ reproduces degree d* + 1, i.e. the next class at ratio 0 (P&L §4.2)
    lc_inf = lpf(; endpoints=MinimumRevision(; degree=0, ratio=Inf)).end_weights
    @test lc_inf == lpf(; endpoints=MinimumRevision(; degree=1, ratio=0)).end_weights
    @test lpf(; endpoints=MinimumRevision(; degree=0, ratio=1e8)).end_weights ≈ lc_inf atol =
        1e-5
end

@testset "estimated ratio (MSRE)" begin
    f = lpf()
    @test f.endpoints isa MinimumRevision
    @test f.endpoints.degree == 1
    @test 0 ≤ f.endpoints.ratio ≤ Inf

    h = 6
    final = predict(lpf(; endpoints=NoEndpoints()), _y)
    function msre(ratio)
        v = lpf(; endpoints=MinimumRevision(; degree=1, ratio)).end_weights[1]
        return sum(abs2(final[t] - dot(v, _y[(t - h):t])) for t in (h + 1):(120 - h))
    end
    for ratio in (0.0, 1e-3, 1e-2, 1e-1, 1.0, Inf)
        @test msre(f.endpoints.ratio) ≤ msre(ratio) + 1e-10
    end

    # Too short to estimate: falls back to ratio 0 and says so
    short = fit(LocalPolynomialFilter, _y[1:20])
    @test short.endpoints.ratio == 0
    @test haskey(short.estimated, :ratio_note)
end

@testset "reporting estimated hyperparameters" begin
    @test_logs (:info, r"LocalPolynomialFilter") lpf()
    @test_logs min_level = Logging.Info lpf(; endpoints=MinimumRevision(; ratio=0.1))
    logs, f = Test.collect_test_logs(() -> lpf(; window=[9, 13]))
    @test length(logs) == 1
    @test logs[1].group === :estimated
    @test logs[1].kwargs[:window] == f.window
    @test logs[1].kwargs[:ratio] == f.endpoints.ratio
    @test haskey(f.estimated, :cv_score) && haskey(f.estimated, :msre)
    @test isempty(lpf(; endpoints=musgrave(3.5)).estimated)
end

@testset "window selection (CV)" begin
    f = lpf(; window=7:2:51, endpoints=NoEndpoints())
    @test isodd(f.window)
    @test f.window ∈ 7:2:51
    @test f.estimated.window == f.window
    @test lpf(; window=[9, 13], endpoints=NoEndpoints()).window ∈ (9, 13)
end

@testset "LocalPolynomialFilter predict" begin
    t = 1:40
    cubic = @. 0.5 + 0.1t - 0.01t^2 + 0.0003t^3
    f = fit(LocalPolynomialFilter, cubic; endpoints=DirectAsymmetric())
    @test predict(f, cubic) ≈ cubic
    linear = @. 2 - 0.3t
    for degree in 0:2
        g = fit(
            LocalPolynomialFilter, linear; endpoints=MinimumRevision(; degree, ratio=0.1)
        )
        degree ≥ 1 && @test predict(g, linear) ≈ linear
    end

    f = lpf()
    @test predict(f, reverse(_y)) ≈ reverse(predict(f, _y))    # mirrored start
    @test length(predict(f, randn(13))) == 13
    @test_throws ArgumentError predict(f, randn(12))
    x = predict(lpf(; endpoints=NoEndpoints()), _y)
    @test all(isnan, x[[1:6; 115:120]])
    @test all(isfinite, x[7:114])
end

@testset "LocalPolynomialFilter validation" begin
    @test_throws ArgumentError lpf(; window=12)
    @test_throws ArgumentError lpf(; window=5, endpoints=DirectAsymmetric())    # h < d
    @test_throws ArgumentError lpf(; endpoints=MinimumRevision(; degree=3, ratio=0.1))
    @test_throws ArgumentError lpf(;
        window=5, endpoints=MinimumRevision(; degree=2, ratio=0.1)
    )
    @test_throws ArgumentError lpf(;
        window=7, endpoints=MinimumRevision(; degree=2, ratio=Inf)
    )
    # Estimated ratio without the Inf candidate
    f = lpf(; window=7, degree=4, endpoints=MinimumRevision(; degree=2))
    @test isfinite(f.endpoints.ratio)
    @test all(isfinite, predict(f, _y))
    @test_throws ArgumentError fit(LocalPolynomialFilter, randn(60); window=3, degree=2)
    @test_throws DomainError MinimumRevision(; ratio=-1)
    @test_throws DomainError musgrave(0)
    @test_throws ArgumentError fit(LocalPolynomialFilter, [1.0, NaN, 3.0])
end

@testset "MovingAverage" begin
    @test fit(MovingAverage, _y; window=5).weights ≈ fill(0.2, 5)
    @test fit(MovingAverage, _y; window=12).weights ≈ [0.5; ones(11); 0.5] ./ 12
    f = fit(MovingAverage, _y; window=3)
    x = predict(f, [1.0, 2.0, 6.0, 4.0])
    @test x ≈ [1.5, 3.0, 4.0, 5.0]
    @test predict(fit(MovingAverage, _y; window=12), fill(2.0, 30)) ≈ fill(2.0, 30)
end

@testset "seasonal filters" begin
    s = repeat([1.0, -2.0, 0.5, 0.5], 6)
    @test predict(fit(StableSeasonalFilter, s; period=4), s) ≈ s
    @test predict(fit(SeasonalMovingAverage, s; period=4), s) ≈ s
    y = s .+ 0.1randn(24)
    x = predict(fit(StableSeasonalFilter, y; period=4), y)
    @test x[1] ≈ sum(y[1:4:24]) / 6
    @test_throws ArgumentError fit(SeasonalMovingAverage, s; period=4, n=2)
    @test_throws DomainError fit(StableSeasonalFilter, s; period=1)
end

@testset "LoessFilter" begin
    f = fit(LoessFilter, _y; window=25)
    x = predict(f, _y)
    @test length(x) == 120
    @test all(isfinite, x)
    @test_throws DomainError fit(LoessFilter, _y; window=25, degree=3)
end
