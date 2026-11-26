using TimeSeriesDecompositions
using DelimitedFiles
using LinearAlgebra: dot
using Logging
using Random: shuffle
using Statistics: mean
using Test

# Henderson weights in closed form (X-11; m = h + 2)
function henderson_closed_form(h)
    m = h + 2
    return map((-h):h) do j
        num = 315 * ((m - 1)^2 - j^2) * (m^2 - j^2) * ((m + 1)^2 - j^2) *
              (3m^2 - 16 - 11j^2)
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

function lpf(;
        y = _ys, bandwidth = 12, degree = 3, kernel = HendersonKernel(),
        endpoints = MinimumRevision()
)
    return fit(LocalPolynomial(; bandwidth, degree, kernel, endpoints), y)
end

loess(y, t = eachindex(y); wts = nothing, kwargs...) = fit(LOESS(; kwargs...), y, t; wts)

@testset "kernel_weights" begin
    @test kernel_weights(HendersonKernel(), 6) ==
          [(49 - j^2) * (64 - j^2) * (81 - j^2) for j in -6:6]
    for k in (
        HendersonKernel(),
        BiweightKernel(),
        TricubeKernel(),
        EpanechnikovKernel(),
        UniformKernel()
    )
        κ = kernel_weights(k, 4)
        @test κ == reverse(κ)
        @test all(>(0), κ)
        @test lpf(; kernel = k, endpoints = NoEndpoints()).state.weights |> sum ≈ 1
    end
end

@testset "Henderson symmetric weights" begin
    rng = StableRNG(301)
    # Published 13-term weights, j = 0:6
    f = lpf(; bandwidth = 12, endpoints = NoEndpoints())
    @test f.state.weights[7:13] ≈
          [0.240057, 0.214337, 0.147357, 0.065492, 0.0, -0.027864, -0.019350] atol = 1e-6
    for bandwidth in (4, 6, 8, 12, 22, 50, 100)
        h = bandwidth ÷ 2
        y = randn(rng, max(bandwidth + 1, 120))
        w = lpf(; y = y, bandwidth, endpoints = NoEndpoints()).state.weights
        @test w ≈ henderson_closed_form(h) atol = 1e-10
        @test sum(w) ≈ 1
    end
    @test lpf(; bandwidth = 8, endpoints = NoEndpoints()).state.weights[5:9] ≈
          [0.331139, 0.266557, 0.118470, -0.009872, -0.040724] atol = 1e-6
end

@testset "direct asymmetric weights (P&L Table 1)" begin
    # Weight on the current observation, h = 6; rows q = 0:6, columns d = 0:6
    table = [0.245665 0.585646 0.835625 0.955184 0.992471 0.999412 1.000000
             0.199086 0.303839 0.306032 0.455995 0.728452 0.923773 0.990781
             0.171191 0.200760 0.265311 0.427533 0.449289 0.518942 0.766170
             0.154679 0.161543 0.265235 0.338531 0.360310 0.514373 0.539702
             0.145554 0.146624 0.257785 0.277607 0.357651 0.430913 0.459352
             0.141309 0.141375 0.247161 0.249482 0.351565 0.364447 0.459300
             0.140033 0.140033 0.240057 0.240057 0.337907 0.337907 0.441782]
    for d in 0:6
        f = lpf(; bandwidth = 12, degree = d, endpoints = DirectAsymmetric())
        for q in 0:5
            @test f.state.end_weights[q + 1][7] ≈ table[q + 1, d + 1] atol = 1e-6
        end
        @test f.state.weights[7] ≈ table[7, d + 1] atol = 1e-6
    end
    @test lpf(; endpoints = DirectAsymmetric()).state.end_weights[1] ≈
          [-0.017237, 0.021887, 0.040002, -0.034147, -0.097894, 0.132204, 0.955184] atol = 1e-6
end

@testset "Musgrave end weights" begin
    f = lpf(; bandwidth = 12, endpoints = musgrave(3.5))
    @test f.model.endpoints.ratio ≈ 0.103938 atol = 1e-6
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
            0.148101
        ]
    ]
    for q in 0:2
        @test f.state.end_weights[q + 1] ≈ published[q + 1] atol = 1e-6
    end
    @test lpf(; bandwidth = 8, endpoints = musgrave(1.0)).state.end_weights[1] ≈
          [-0.155536, -0.033836, 0.185356, 0.424292, 0.579724] atol = 1e-6
    for (bandwidth, R) in ((8, 1.0), (12, 3.5), (22, 4.5))
        f = lpf(; bandwidth, endpoints = musgrave(R))
        h = bandwidth ÷ 2
        for q in 0:(h - 1)
            @test f.state.end_weights[q + 1] ≈
                  musgrave_closed_form(f.state.weights, h + 1 + q, R) atol = 1e-12
        end
    end
    @test musgrave_ic_ratio(9) == 1.0
    @test musgrave_ic_ratio(13) == 3.5
    @test musgrave_ic_ratio(23) == 4.5
    @test_throws ArgumentError musgrave_ic_ratio(15)
end

@testset "minimum-revision weights" begin
    ql = lpf(; endpoints = MinimumRevision(; degree = 1, ratio = 0.016))
    @test ql.state.end_weights[1] ≈
          [0.021586, -0.087155, -0.096713, -0.005838, 0.164216, 0.382191, 0.621712] atol = 1e-6
    cq = lpf(; endpoints = MinimumRevision(; degree = 2, ratio = 0.003))
    @test cq.state.end_weights[1] ≈
          [0.064348, -0.013092, -0.092745, -0.099298, 0.020058, 0.308128, 0.812601] atol = 1e-6

    # Reproduction constraints Uₚᵀv = Uᵀw for polynomials of degree d*
    h, js = 6, -6:6
    for degree in 0:2, q in 0:(h - 1)

        f = lpf(; endpoints = MinimumRevision(; degree, ratio = 0.05))
        v, p = f.state.end_weights[q + 1], 1:(h + 1 + q)
        for k in 0:degree
            @test dot(v, js[p] .^ k) ≈ dot(f.state.weights, js .^ k) atol = 1e-9
        end
    end

    # ratio → ∞ reproduces degree d* + 1, i.e. the next class at ratio 0 (P&L §4.2)
    lc_inf = lpf(; endpoints = MinimumRevision(; degree = 0, ratio = Inf)).state.end_weights
    @test lc_inf ==
          lpf(; endpoints = MinimumRevision(; degree = 1, ratio = 0)).state.end_weights
    @test lpf(; endpoints = MinimumRevision(; degree = 0, ratio = 1e8)).state.end_weights ≈
          lc_inf atol = 1e-5
end

@testset "estimated ratio (MSRE)" begin
    f = lpf()
    @test f.model.endpoints isa MinimumRevision
    @test f.model.endpoints.degree == 1
    @test 0 ≤ f.model.endpoints.ratio ≤ Inf

    h = 6
    final = predict(lpf(; endpoints = NoEndpoints()))
    function msre(ratio)
        v = lpf(; endpoints = MinimumRevision(; degree = 1, ratio)).state.end_weights[1]
        return sum(abs2(final[t] - dot(v, _ys[(t - h):t])) for t in (h + 1):(120 - h))
    end
    for ratio in (0.0, 1e-3, 1e-2, 1e-1, 1.0, Inf)
        @test msre(f.model.endpoints.ratio) ≤ msre(ratio) + 1e-10
    end

    # Too short to estimate: falls back to ratio 0 and says so
    short = lpf(; y = _ys[1:20])
    @test short.model.endpoints.ratio == 0
    @test haskey(short.estimated, :note_ratio)
end

@testset "reporting estimated hyperparameters" begin
    @test_logs (:info, r"^Henderson") lpf()
    @test_logs min_level = Logging.Info lpf(; endpoints = MinimumRevision(; ratio = 0.1))
    logs, f = Test.collect_test_logs(() -> lpf(; bandwidth = [8, 12]))
    @test length(logs) == 1
    @test logs[1].group === :estimated
    @test logs[1].kwargs[:bandwidth] == f.model.bandwidth
    @test logs[1].kwargs[:ratio] == f.model.endpoints.ratio
    @test haskey(f.estimated, :cv_score) && haskey(f.estimated, :msre)
    @test isempty(lpf(; endpoints = musgrave(3.5)).estimated)
end

@testset "bandwidth selection (CV)" begin
    f = lpf(; bandwidth = 6:2:50, endpoints = NoEndpoints())
    @test iseven(f.model.bandwidth)
    @test f.model.bandwidth ∈ 6:2:50
    @test f.estimated.bandwidth == f.model.bandwidth
    @test lpf(; bandwidth = [8, 12], endpoints = NoEndpoints()).model.bandwidth ∈ (8, 12)
    # in units of t: the same filter on a grid of spacing 0.5
    g = fit(LocalPolynomial(; bandwidth = 6, endpoints = NoEndpoints()), _ys, 0.5 .*
                                                                              (1:120))
    @test g.state.weights ≈ lpf(; bandwidth = 12, endpoints = NoEndpoints()).state.weights
    @test_throws ArgumentError lpf(; bandwidth = 11)    # odd multiple of Δ
    @test_throws DomainError LocalPolynomial(; bandwidth = 0)
end

@testset "bandwidth selection scores all points with the end weights" begin
    GP = TimeSeriesDecompositions
    N, degree = length(_ys), 3
    for endpoints in (DirectAsymmetric(), musgrave(3.5), MinimumRevision())
        f = @quietly lpf(; bandwidth = [8, 12, 16], endpoints)
        # the score of the chosen filter: closed-form leave-one-out over all N points
        values, self = GP._filter_values(f.state.weights, f.state.end_weights, _ys)
        @test values ≈ predict(f)
        @test f.estimated.cv_score ≈ mean(@. ((_ys - values) / (1 - self))^2)
    end
    # one-sided local cubic at the last point: the weighted fit without that point
    h = 6
    g = GP._grid_kernel(HendersonKernel(), h)
    _, ends = GP._filter_weights(g, DirectAsymmetric(), degree)
    v = ends[1]    # offsets -h:0 at t = N
    d = collect((-h):-1)
    X = [k^j for k in d, j in 0:degree]
    W = [g.κ(k) for k in d]
    β = (X' * (W .* X)) \ (X' * (W .* _ys[N .+ d]))
    @test (_ys[N] - dot(v, _ys[(N - h):N])) / (1 - v[h + 1]) ≈ _ys[N] - β[1]
end

@testset "LocalPolynomial predict" begin
    rng = StableRNG(302)
    t = 1:40
    cubic = @. 0.5 + 0.1t - 0.01t^2 + 0.0003t^3
    f = lpf(; y = cubic, endpoints = DirectAsymmetric())
    @test predict(f) ≈ cubic
    linear = @. 2 - 0.3t
    for degree in 0:2
        g = lpf(; y = linear, endpoints = MinimumRevision(; degree, ratio = 0.1))
        degree ≥ 1 && @test predict(g) ≈ linear
    end

    f = lpf()
    @test predict(fit(f.model, reverse(_ys))) ≈ reverse(predict(f))    # mirrored start
    @test length(predict(fit(f.model, randn(rng, 13)))) == 13
    @test_throws ArgumentError fit(f.model, randn(rng, 12))
    x = predict(lpf(; endpoints = NoEndpoints()))
    @test all(isnan, x[[1:6; 115:120]])
    @test all(isfinite, x[7:114])
    @test fit(Henderson, _ys; bandwidth = 12).model ==
          Henderson(; bandwidth = 12) |> m -> fit(m, _ys).model
end

@testset "type stability and defaults" begin
    t = collect(1.0:120)
    for model in (
        Henderson(; bandwidth = 12),
        Henderson(; bandwidth = 12, endpoints = musgrave(3.5)),
        MovingAverage(; bandwidth = 12),
        LOESS(; span = 0.3)
    )
        f = @inferred TimeSeriesDecompositions._fit(model, _ys, t)
        @test @inferred(predict(f, [10.5, 60.25])) isa Vector{Float64}
    end
    # a bandwidth left as nothing is chosen from 2h = 4 up to h = (N - 1) ÷ 4
    f = @quietly fit(Henderson(; endpoints = NoEndpoints()), _ys)
    @test f.model.bandwidth ∈ 4:2:(2 * ((length(_ys) - 1) ÷ 4))
    @test haskey(f.estimated, :cv_score) && haskey(f.estimated, :autocorrelation_z)
    @test_logs (:info, r"^Henderson") match_mode = :any fit(Henderson(), _ys)
end

@testset "LocalPolynomial between grid points" begin
    rng = StableRNG(303)
    t = 1:40
    p(x) = 0.5 + 0.1x - 0.01x^2 + 0.0003x^3
    xs = [1.3, 3.5, 20.25, 37.5, 39.9]
    f = lpf(; y = p.(t), endpoints = DirectAsymmetric())
    @test predict(f, xs) ≈ p.(xs)    # local cubic reproduces cubics everywhere
    @test predict(f, [5.0, 20.0]) == predict(f)[[5, 20]]
    q(x) = 2 - 0.3x
    g = lpf(; y = q.(t), endpoints = MinimumRevision(; ratio = 0.05))
    @test predict(g, xs) ≈ q.(xs)    # QL reproduces lines near the ends too
    # Interior: continuous in x, reducing to the grid weights at grid points
    h = lpf(; endpoints = NoEndpoints())
    @test predict(h, [60.0 + 1e-9, 60.0 - 1e-9]) ≈ predict(h)[[60, 60]] atol = 1e-6
    @test predict(h, [60.5]) ≈ predict(h, [60.5 + 1e-7]) atol = 1e-6
    @test_throws DomainError predict(h, [0.5])
    u = lpf(; degree = 1, kernel = UniformKernel(), endpoints = DirectAsymmetric())
    @test predict(u, [12 - 1e-7]) ≈ predict(u, [12 + 1e-7]) atol = 1e-5
    # Near the ends, minimum-revision estimates are interpolated: continuous, exact at
    # grid points and meeting the symmetric part continuously
    m = lpf(; endpoints = musgrave(3.5))
    for x in (120.0, 114.0, 1.0, 7.0)
        @test predict(m, [x - 1e-7, x + 1e-7] .|> z -> clamp(z, 1, 120)) ≈
              fill(predict(m)[Int(x)], 2) atol = 1e-5
    end
    @test_throws ArgumentError fit(Henderson(; bandwidth = 12), _ys, sort(rand(rng, 120)))
end

@testset "LocalPolynomial validation" begin
    rng = StableRNG(304)
    @test_throws ArgumentError lpf(; bandwidth = 11)
    @test_throws ArgumentError lpf(; bandwidth = 4, endpoints = DirectAsymmetric())    # h < d
    @test_throws ArgumentError lpf(; endpoints = MinimumRevision(; degree = 3, ratio = 0.1))
    @test_throws ArgumentError lpf(;
        bandwidth = 4, endpoints = MinimumRevision(; degree = 2, ratio = 0.1)
    )
    @test_throws ArgumentError lpf(;
        bandwidth = 6, endpoints = MinimumRevision(; degree = 2, ratio = Inf)
    )
    # Estimated ratio without the Inf candidate
    f = lpf(; bandwidth = 6, degree = 4, endpoints = MinimumRevision(; degree = 2))
    @test isfinite(f.model.endpoints.ratio)
    @test all(isfinite, predict(f))
    @test_throws ArgumentError lpf(; y = randn(rng, 60), bandwidth = 2, degree = 2)
    @test_throws DomainError MinimumRevision(; ratio = -1)
    @test_throws DomainError musgrave(0)
    @test_throws ArgumentError lpf(; y = [1.0, NaN, 3.0])
end

@testset "MovingAverage on a grid" begin
    rng = StableRNG(305)
    x = collect(1.0:30)
    # Odd bandwidth: the simple average; even bandwidth: the centered 2 × m average
    @test predict(fit(MovingAverage(; bandwidth = 5), x))[3:28] ≈ x[3:28]
    y = randn(rng, 40)
    ma = predict(fit(MovingAverage(; bandwidth = 12), y))
    @test ma[7:34] ≈ [dot([0.5; ones(11); 0.5] ./ 12, y[(i - 6):(i + 6)]) for i in 7:34]
    @test all(isnan, ma[[1:6; 35:40]])
    # Annihilates a fixed seasonal of the same period, also between grid points
    seasonal = [sin(2π * i / 12) + 0.3cos(4π * i / 12) for i in 1:60]
    f = fit(MovingAverage(; bandwidth = 12), seasonal)
    @test maximum(abs, filter(isfinite, predict(f))) < 1e-12
    @test maximum(abs, predict(f, [20.5, 30.25, 41.7])) < 1e-12
    # End weights on a grid: truncated average (DirectAsymmetric) or minimum revision
    trunc = predict(fit(MovingAverage(; bandwidth = 3, endpoints = DirectAsymmetric()), x[1:4]))
    @test trunc ≈ [1.5, 2.0, 3.0, 3.5]
    mr = fit(MovingAverage(; bandwidth = 12, endpoints = MinimumRevision(; degree = 0)), y)
    @test all(isfinite, predict(mr))
    for x in (6.0, 6.5, 7.0, 34.0, 34.5, 35.0)    # continuous where end weights take over
        @test predict(mr, [x - 1e-7]) ≈ predict(mr, [x + 1e-7]) atol = 1e-5
    end
    @test_throws ArgumentError fit(MovingAverage(; bandwidth = 12), y, sort(rand(rng, 40)) .*
                                                                       40;) |>
                               f -> fit(MovingAverage(; bandwidth = 12, endpoints = DirectAsymmetric()), y, f.t)
    @test_throws ArgumentError fit(MovingAverage(), y)
end

@testset "MovingAverage at irregular times" begin
    rng = StableRNG(3)
    t = sort((1:240) .+ 0.4 .* (2 .* rand(rng, 240) .- 1))    # jittered grid
    seasonal = sin.(2π .* t ./ 12)
    f = fit(MovingAverage(; bandwidth = 12), seasonal, t)
    @test sqrt(mean(abs2, filter(isfinite, predict(f)))) < 0.03
    # Reproduces constants; tied times share one cell
    @test all(
        ≈(2.0), filter(isfinite, predict(fit(MovingAverage(; bandwidth = 12), fill(2.0, 240), t)))
    )
    tied = [1.0, 2, 2, 3, 4, 5, 6, 7, 8]
    g = fit(MovingAverage(; bandwidth = 3), [1.0, 2, 4, 3, 4, 5, 6, 7, 8], tied)
    @test predict(g, [3.0]) ≈ [(3 + 3 + 4) / 3]
    # A gap wider than the window gives NaN with a warning
    gappy = [1.0:50; 80.0:130]
    logs, h = Test.collect_test_logs(() -> fit(MovingAverage(; bandwidth = 6), randn(rng, 101), gappy))
    @test any(l -> l.level == Logging.Warn && occursin("gap", l.message), logs)
    @test isnan(predict(h, [57.0])[1])    # the window lies in the cell of t = 50
    @test isfinite(predict(h, [25.0])[1])
    # Bursts of close observations do not shrink the typical spacing
    bursty = sort([0.0:3:120; (0:9) .* 0.05 .+ 30; (0:9) .* 0.05 .+ 90])
    b = fit(MovingAverage(; bandwidth = 12), sin.(2π .* bursty ./ 12), bursty)
    @test count(isnan, predict(b)) < 10
end

@testset "seasonal smoothers" begin
    rng = StableRNG(308)
    s = repeat([1.0, -2.0, 0.5, 0.5], 6)
    @test predict(fit(StableSeasonal(), s; period = 4)) ≈ s
    @test predict(fit(SeasonalMovingAverage(), s; period = 4)) ≈ s
    @test predict(fit(SeasonalLOESS(), s; period = 4)) ≈ s
    # the window of a seasonal loess: an even number of periods, six by default
    @test fit(SeasonalLOESS(), s; period = 4).model.bandwidth == 24
    @test fit(SeasonalLOESS(; bandwidth = 9), s; period = 4).model.bandwidth == 16
    @test sprint(show, SeasonalLOESS()) == "SeasonalLOESS(bandwidth=nothing, degree=1)"
    y = s .+ 0.1randn(rng, 24)
    f = fit(StableSeasonal(), y; period = 4)
    @test predict(f)[1] ≈ sum(y[1:4:24]) / 6
    @test predict(f, [5.0, 5.0 + 1e-12]) == predict(f)[[5, 5]]
    # Between the samples: a sinusoid is interpolated exactly, for odd and even periods
    for p in (3, 4, 12)
        x = 1:(6p)
        g = fit(StableSeasonal(), cospi.(2 .* x ./ p .+ 0.2), x; period = p)
        τ = range(1, 6p; length = 101)
        @test predict(g, τ) ≈ cospi.(2 .* τ ./ p .+ 0.2) atol = 1e-12
    end
    @test_throws DomainError predict(f, [24.5])
    @test_throws ArgumentError SeasonalMovingAverage(; n = 2)
    @test_throws DomainError SeasonalLOESS(; bandwidth = 0)
    @test_throws DomainError SeasonalLOESS(; degree = 3)
    @test_throws ArgumentError fit(SeasonalLOESS(; bandwidth = 8), s; period = 4)
    @test predict(fit(SeasonalLOESS(; bandwidth = 8, degree = 0), s; period = 4)) ≈ s
    @test_throws ArgumentError fit(StableSeasonal(), s; period = 1)
end

# Direct weighted least-squares loess with bandwidth λ at x, for reference
function loess_reference(t, y, λ, degree, x; wts = ones(length(t)))
    w = wts .* (1 .- min.(abs.(t .- x) ./ λ, 1) .^ 3) .^ 3
    U = [((tᵢ - x) / λ)^k for tᵢ in t, k in 0:degree]
    return ((U' * (w .* U)) \ (U' * (w .* y)))[1]
end

@testset "LOESS matches R loess (surface = \"direct\")" begin
    data = readdlm(joinpath(@__DIR__, "data", "loess_r_data.txt"))
    fits = readdlm(joinpath(@__DIR__, "data", "loess_r_fits.txt"))
    t, y, w = data[:, 1], data[:, 2], data[:, 3]
    tn = range(minimum(t), maximum(t); length = 17)
    for row in eachrow(fits)
        weighted, degree, span, enp = row[1] == 1, Int(row[2]), row[3], row[4]
        wts = weighted ? w : ones(50)
        s = loess(y, t; span, degree, wts)
        @test predict(s, tn) ≈ row[5:21] atol = 1e-10
        @test predict(s) ≈ row[22:71] atol = 1e-10
        L = TimeSeriesDecompositions._hat_matrix(s.t, s.state.wts, span, 0.0, degree)
        @test sum(L[i, i] for i in 1:50) ≈ enp atol = 1e-10
    end
end

@testset "LOESS bandwidth and weights" begin
    rng = StableRNG(306)
    t = sort(rand(rng, 60) .* 20)
    y = sin.(t) .+ 0.2randn(rng, 60)
    tn = range(t[1], t[end]; length = 13)
    s = loess(y, t; bandwidth = 6.0, degree = 1)    # twice the radius λ = 3
    @test predict(s, tn) ≈ [loess_reference(t, y, 3.0, 1, x) for x in tn]
    # The larger of the span and fixed bandwidths
    both = loess(y, t; span = 0.1, bandwidth = 4.0, degree = 2)
    q = TimeSeriesDecompositions._span_neighbors(0.1, 60)
    λ(x) = max(2.0, sort(abs.(t .- x))[q])
    @test predict(both, tn) ≈ [loess_reference(t, y, λ(x), 2, x) for x in tn]
    # Span above one: √α times the distance to the farthest observation, as in R
    wide = loess(y, t; span = 2.0, bandwidth = 2.0, degree = 1)
    λwide(x) = max(1.0, √2 * maximum(abs.(t .- x)))
    @test predict(wide, tn) ≈ [loess_reference(t, y, λwide(x), 1, x) for x in tn]
    @test_throws ArgumentError loess([1.0, 2, 3, 4], [0, 1, 10, 11]; bandwidth = 4)

    wts = rand(rng, 60)
    s = loess(y, t; span = 0.3, degree = 1, wts)
    x = t[20]
    λq = sort(abs.(t .- x))[TimeSeriesDecompositions._span_neighbors(0.3, 60)]
    @test predict(s, [x]) ≈ [loess_reference(t, y, λq, 1, x; wts)]
end

@testset "LOESS on a grid equals the tricube LocalPolynomial" begin
    for degree in 1:2, q in (9, 15, 21)

        s = loess(_ys; span = q / 120, degree)
        f = lpf(; bandwidth = q - 3, degree, kernel = TricubeKernel(), endpoints = NoEndpoints())
        inner = ((q + 1) ÷ 2):(120 - (q - 1) ÷ 2)    # symmetric neighborhoods
        @test predict(s)[inner] ≈ predict(f)[inner] atol = 1e-12
    end
end

@testset "LOESS interpolation and reproduction" begin
    rng = StableRNG(307)
    t = sort(rand(rng, 40) .* 10)
    tn = sort(rand(rng, 25) .* (t[end] - t[1]) .+ t[1])
    for degree in 0:2
        p(x) = evalpoly(x, [1.0, -0.5, 0.2][1:(degree + 1)])
        for kwargs in ((; span = 0.25), (; bandwidth = 5.0), (;
            span = 0.25, bandwidth = 2.0))
            s = loess(p.(t), t; degree, kwargs...)
            @test predict(s, tn) ≈ p.(tn)
            @test predict(s) ≈ p.(s.t)
        end
    end
    s = loess(_ys; span = 0.15)
    @test predict(s, s.t) == predict(s)
    @test fitted(s) == predict(s)
    @test residuals(s) ≈ s.y - predict(s)
    @test_throws DomainError predict(s, [0.0])
    @test length(predict(s, [0.0, 121.0]; extrapolate = true)) == 2
    perm = shuffle(StableRNG(310), 1:120)    # unsorted input is sorted by t
    @test predict(loess(_ys[perm], perm; span = 0.15)) ≈ predict(s)
end

@testset "LOESS validation" begin
    @test_throws DomainError loess(_ys; span = 0)
    @test_throws DomainError loess(_ys; bandwidth = -1)
    @test_throws ArgumentError loess(_ys; span = 0.02)    # q < degree + 2
    @test_throws DomainError loess(_ys; span = 0.5, degree = 3)
    @test_throws ArgumentError loess(_ys; span = 0.5, wts = (-ones(120)))
    @test_throws DimensionMismatch loess(_ys, 1:10; span = 0.5)
    @test_throws ArgumentError loess(
        [1.0, 2.0, 3.0, 4.0], [1, 1, 1, 2]; span = 0.75, degree = 1
    )    # three observations tied at t = 1
end

@testset "LOESS span selection (leave-one-out)" begin
    logs, s = Test.collect_test_logs(() -> loess(_ys))
    @test length(logs) == 1
    @test logs[1].group === :estimated
    @test s.estimated.span == s.model.span
    @test haskey(s.estimated, :enp) && haskey(s.estimated, :cv_score)
    # candidate spans, scored by the same criterion
    c = loess(_ys; span = [0.1, s.model.span, 0.9])
    @test c.model.span == s.model.span
    @test c.estimated.cv_score ≈ s.estimated.cv_score
    @test_logs min_level = Logging.Info loess(_ys; span = 0.3)
    @test_logs min_level = Logging.Info loess(_ys; bandwidth = 10)

    # Spans too small for tied times are skipped rather than failing
    rng = StableRNG(309)
    for replicates in (3, 5)
        tied = repeat(1.0:50; inner = replicates)
        s = loess(sin.(tied ./ 5) .+ 0.1randn(rng, length(tied)), tied)
        @test isfinite(s.estimated.cv_score)
    end

    # Warns about autocorrelated errors, but not for independent ones
    rng = StableRNG(11)
    ar = zeros(500)
    for i in 2:500
        ar[i] = 0.5ar[i - 1] + randn(rng)
    end
    @test_logs (:warn, r"autocorrelated") match_mode = :any loess(ar)
    iid = [sin(2π * i / 100) for i in 1:500] .+ randn(rng, 500)
    @test_logs min_level = Logging.Warn loess(iid)
end

@testset "closed-form leave-one-out residuals" begin
    rng = StableRNG(5)
    t = collect(1.0:60)
    y = sin.(t ./ 8) .+ 0.2randn(rng, 60)
    # loess with a fixed window: refitting without observation i
    f, lev = TimeSeriesDecompositions._loess_fit(t, y, ones(60), 0.0, 6.0, 2)
    for i in (1, 17, 30, 60)
        keep = setdiff(1:60, i)
        refit = predict(
            fit(LOESS(; bandwidth = 12), y[keep], t[keep]), t[[i]]; extrapolate = true)[1]
        @test (y[i] - f[i]) / (1 - lev[i]) ≈ y[i] - refit
    end
    # a symmetric local cubic: the weighted fit at t without t
    h = 5
    g = TimeSeriesDecompositions._grid_kernel(HendersonKernel(), h)
    w = TimeSeriesDecompositions._local_weights(g, (-h):h, 3)
    d = [k for k in (-h):h if k != 0]
    X = [k^j for k in d, j in 0:3]
    W = [g.κ(k) for k in d]
    i = 30
    β = (X' * (W .* X)) \ (X' * (W .* y[i .+ d]))
    @test (y[i] - dot(w, y[(i - h):(i + h)])) / (1 - w[h + 1]) ≈ y[i] - β[1]
end
