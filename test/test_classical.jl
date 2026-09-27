using GPDecompositions
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
    @test_throws ArgumentError fit(D, _yc, [_tc[1:(end - 1)]; _tc[end] + 1]; period=1.0)   # non-uniform
    @test_throws ArgumentError fit(D, _yc, _tc; period=0.1)                              # non-integer T
    @test_throws DomainError fit(D, _yc, _tc; period=-1.0)
    @test_throws ArgumentError fit(D, _yc, _tc; period=1.0, trend_window=12)             # even
    @test_throws ArgumentError fit(D, _yc[1:20], _tc[1:20]; period=1.0)                  # too short
    @test_throws DimensionMismatch fit(D, _yc, _tc[1:(end - 1)]; period=1.0)
end

@testset "X11Decomp options" begin
    d = fit(X11Decomp, _yc, _tc; period=1.0)
    f = d.θ.filters.trend
    @test f isa LocalPolynomialFilter
    @test f.window == 13
    @test f.endpoints.ratio ≈ 4 / (π * 3.5^2)
    @test fit(X11Decomp, _yc, _tc; period=1.0, endpoints=DirectAsymmetric()).trend.mean !=
        d.trend.mean
    d23 = fit(X11Decomp, _yc, _tc; period=1.0, trend_window=23)
    @test d23.θ.filters.trend.endpoints.ratio ≈ 4 / (π * 4.5^2)
    dh = fit(X11Decomp, _yc, _tc; period=1.0, endpoints=nothing)
    @test all(isfinite, dh.trend.mean[7:(end - 6)])
    @test all(isnan, dh.trend.mean[[1:6; (end - 5):end]])
    @test all(isfinite, dh.rate.mean[7:(end - 6)])
    @test_throws ArgumentError fit(
        X11Decomp, _yc, _tc; period=1.0, seasonal_filters=((3, 2), (3, 5))
    )
end

@testset "STLDecomp default trend_window" begin
    d = fit(STLDecomp, _yc, _tc; period=1.0)
    @test isodd(d.θ.filters.trend.window)
    @test d.θ.filters.trend.window ≥ 1.5 * 12 / (1 - 1.5 / 7)
end
