using GPDecompositions
using Test

const _t = collect(1.0:50.0)
const _y = sin.(2π .* _t ./ 12) .+ 0.01 .* _t .+ 0.1 .* randn(50)

@testset "model and shorthand" begin
    m = GaussianProcess(; approx_rate = true)
    d = fit(m, _y, _t; period = 12.0)
    @test d isa Decomposition{<:GaussianProcess}
    @test d.model === m
    @test d.period == 12.0
    @test fit(GaussianProcess, _y, _t; period = 12.0, approx_rate = true).θ == d.θ
    @test m.kernel == (trend = Matern52Kernel(),
        seasonal = (effect = Matern52Kernel(), envelope = Matern52Kernel()))
    @test sprint(show, GaussianProcess()) == "GaussianProcess()"
    @test sprint(show, m) == "GaussianProcess(approx_rate=true)"
    # settings are checked when the model is built
    @test_throws ArgumentError GaussianProcess(; θ_init = (lengthscale = 1.0,))
    @test_throws ArgumentError GaussianProcess(; θ_fixed = (period = 12.0,))
    @test_throws ArgumentError GaussianProcess(; kernel = (bandwidth = SEKernel(),))
    @test_throws DomainError GaussianProcess(; θ_fixed = (bandwidth = -1.0,))
    @test_throws DomainError GaussianProcess(; max_iter = -1)
    # the positional constructor checks too
    @test_throws ArgumentError GaussianProcess(
        m.kernel, (; noise = 1.0), (;), false, m.optimizer, 10)
end

@testset "Component fields" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0)
    @test !isnothing(d.seasonal)
    for comp in (d.trend, d.seasonal, d.noise)
        @test comp isa Component
        @test length(comp.mean) == 50
        @test length(comp.std) == 50
        @test length(comp.lower) == length(comp.sig)
        @test length(comp.upper) == length(comp.sig)
        @test all(comp.std .≥ 0)
    end
    @test d.rate isa Component
end

@testset "no seasonal component (default)" begin
    d = fit(GaussianProcess, _y, _t)
    @test isnothing(d.seasonal)
    @test d.trend isa Component
    @test d.noise isa Component
    @test d.θ.σ_seasonal == 0.0
    @test d.θ.l_seasonal == Inf
    @test d.θ.bandwidth == Inf
end

@testset "no seasonal component (explicit Inf)" begin
    d = fit(GaussianProcess, _y, _t; period = Inf)
    @test isnothing(d.seasonal)
    @test d.θ.σ_seasonal == 0.0
    @test d.θ.l_seasonal == Inf
end

@testset "bandwidth ignored when no seasonal" begin
    d = fit(GaussianProcess, _y, _t; θ_fixed = (bandwidth = 30.0,))
    @test isnothing(d.seasonal)
    @test d.θ.bandwidth == Inf
end

@testset "rate component (approx, NaN std)" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0, approx_rate = true)
    @test d.rate isa Component
    @test length(d.rate.mean) == 50
    @test all(isnan, d.rate.std)
    @test all(isnan, d.rate.lower[1])
    @test all(isnan, d.rate.upper[1])
end

@testset "rate component (full, finite std)" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0)
    @test d.rate isa Component
    @test length(d.rate.mean) == 50
    @test all(isfinite, d.rate.std)
    @test all(d.rate.std .≥ 0)
end

@testset "sig levels" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0)
    @test d.trend.sig == [0.5, 0.9]
    d = predict(d; sig = [0.68, 0.95])
    @test d.trend.sig == d.seasonal.sig == d.rate.sig == [0.68, 0.95]
    @test length(d.trend.lower) == 2
    @test predict(d; sig = [0.95, 0.5, 0.5]).trend.sig == [0.5, 0.95]
    @test_throws DomainError predict(d; sig = [1.0])
end

@testset "predict with no t_new uses decomp.t" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0)
    d2 = predict(d)
    @test d2.t == d.t
    @test length(d2.trend.mean) == length(_t)
end

@testset "predict at new times" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0)
    t_new = collect(51.0:60.0)
    d2 = predict(d, t_new)
    @test d2 isa Decomposition{<:GaussianProcess}
    @test length(d2.t) == 10
    @test length(d2.trend.mean) == 10
end

@testset "predict preserves sig" begin
    d = predict(fit(GaussianProcess, _y, _t; period = 12.0); sig = [0.9])
    d2 = predict(d, collect(51.0:55.0))
    @test d2.trend.sig == [0.9]
end

@testset "components at the sorted times" begin
    perm = reverse(eachindex(_t))
    d = fit(GaussianProcess, _y[perm], _t[perm]; period = 12.0)
    @test d.t == _t
    @test d.trend.mean ≈ fit(GaussianProcess, _y, _t; period = 12.0).trend.mean rtol = 1e-4
end

@testset "θ NamedTuple has required keys" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0)
    @test haskey(d.θ, :σ_trend)
    @test haskey(d.θ, :l_trend)
    @test haskey(d.θ, :σ_seasonal)
    @test haskey(d.θ, :l_seasonal)
    @test haskey(d.θ, :bandwidth)
    @test haskey(d.θ, :σ_noise)
    @test keys(d.θ) == GPDecompositions._GP_HYPERPARAMETERS
end

@testset "quasi-periodic bandwidth" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0, θ_fixed = (bandwidth = 30.0,))
    @test isfinite(d.θ.bandwidth)
    @test d.θ.bandwidth ≈ 30.0
end

@testset "σ_seasonal=0 disables seasonal" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0, θ_init = (σ_seasonal = 0.0,))
    @test isnothing(d.seasonal)
    @test d.period == Inf
    @test d.θ.σ_seasonal == 0.0
    @test d.θ.l_seasonal == Inf
    @test d.θ.bandwidth == Inf
end

@testset "σ_noise=0 disables noise" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0, θ_init = (σ_noise = 0.0,))
    @test isnothing(d.noise)
    @test d.θ.σ_noise == 0.0
    @test !isnothing(d.seasonal)
end

@testset "custom kernels" begin
    d = fit(
        GaussianProcess,
        _y,
        _t;
        period = 12.0,
        kernel = (trend = Matern52Kernel(), seasonal = Matern32Kernel())
    )
    @test d.trend isa Component
    @test !isnothing(d.seasonal)
end

@testset "kernel kwarg resolution" begin
    R = GPDecompositions._resolve_kernels
    M52, M32, SE = Matern52Kernel(), Matern32Kernel(), SEKernel()
    @test R(M52) == (trend = M52, seasonal = (effect = M52, envelope = M52))
    @test R(SE) == (trend = SE, seasonal = (effect = SE, envelope = SE))
    @test R((trend = SE,)) == (trend = SE, seasonal = (effect = M52, envelope = M52))
    @test R((trend = SE, seasonal = M32)) ==
          (trend = SE, seasonal = (effect = M32, envelope = M32))
    @test R((seasonal = (effect = M32, envelope = SE),)) ==
          (trend = M52, seasonal = (effect = M32, envelope = SE))
    @test R((seasonal = (effect = M32,),)) ==
          (trend = M52, seasonal = (effect = M32, envelope = M52))
    @test_throws ArgumentError R((bandwidth = SE,))
    @test_throws ArgumentError R((seasonal = (bandwidth = SE,),))
    @test_throws ArgumentError R((trend = 1.0,))

    d = fit(
        GaussianProcess,
        _y,
        _t;
        period = 12.0,
        θ_fixed = (bandwidth = 30.0,),
        kernel = (trend = SE, seasonal = (effect = M32, envelope = SE))
    )
    s = sprint(show, MIME"text/plain"(), d)
    @test occursin(r"trend\s+SqExponentialKernel", s)
    @test occursin(r"seasonal\s+Matern32Kernel", s)
    @test occursin(r"envelope\s+SqExponentialKernel", s)
end

@testset "θ from fit can be passed as θ_init" begin
    d1 = fit(GaussianProcess, _y, _t; period = 12.0, θ_fixed = (bandwidth = 30.0,))
    d2 = fit(GaussianProcess, _y, _t; period = 12.0, θ_fixed = d1.θ)    # the same fit
    @test collect(values(d2.θ)) ≈ collect(values(d1.θ))
    @test d2.θ.bandwidth ≈ 30.0
    @test d2.θ.l_trend ≈ d1.θ.l_trend
    @test isempty(d2._fit_info.fit.free)
    @test !isnothing(d2.seasonal)
    # bandwidth re-estimated
    d3 = fit(GaussianProcess, _y, _t; period = 12.0, θ_init = d1.θ)
    @test !isnothing(d3._fit_info.fit.bandwidth_test)
end

@testset "θ_init and θ_fixed" begin
    # a hyperparameter in both is fixed at the θ_fixed value, with a warning
    d = @test_logs (:warn, r"θ_init and θ_fixed both give σ_noise") match_mode = :any fit(
        GaussianProcess, _y, _t; θ_init = (σ_noise = 0.5, l_trend = 20.0),
        θ_fixed = (σ_noise = 0.2,)
    )
    @test d.θ.σ_noise ≈ 0.2
    @test :σ_noise ∉ d._fit_info.fit.free
    @test :l_trend ∈ d._fit_info.fit.free
    @test_throws ArgumentError fit(GaussianProcess, _y, _t; θ_init = (lengthscale = 1.0,))
    @test_throws ArgumentError fit(GaussianProcess, _y, _t; θ_fixed = (noise = 1.0,))
end

@testset "θ_init from a seasonal fit without a period" begin
    d1 = fit(GaussianProcess, _y, _t; period = 12.0)
    d2 = fit(GaussianProcess, _y, _t; θ_init = d1.θ)
    @test isnothing(d2.seasonal)
end

@testset "show" begin
    d = fit(
        GaussianProcess,
        _y,
        _t;
        period = 12.0,
        θ_fixed = (σ_noise = 0.1, bandwidth = 30.0)
    )
    @test sprint(show, d) == "Decomposition{GaussianProcess}(N=50, t∈[1.0, 50.0])"
    s = sprint(show, MIME"text/plain"(), d)
    for label in ("kernels", "estimated", "fixed", "fit", "envelope", "l_trend")
        @test occursin(label, s)
    end
    @test occursin(r"σ_noise\s+\S", s[findfirst("fixed", s)[end]:end])

    d = fit(GaussianProcess, _y, _t; approx_rate = true)
    s = sprint(show, MIME"text/plain"(), d)
    @test occursin("period     : none", s)
    @test occursin("approx_rate", s)
    @test !occursin("σ_seasonal", s)
    @test sprint(show, MIME"text/plain"(), predict(d, 1.0:3.0)) isa String
end

@testset "fields by name" begin
    d = fit(GaussianProcess, _y, _t; period = 12.0, approx_rate = true)
    for name in (:model, :t, :period, :trend, :seasonal, :noise, :rate, :θ)
        @test d[name] === getproperty(d, name)
    end
    for name in (:mean, :std, :lower, :upper, :sig)
        @test d[:trend][name] === getproperty(d.trend, name)
    end
    @test d[:θ][:σ_trend] == d.θ.σ_trend
    @test_throws Exception d[:level]    # FieldError (Julia ≥ 1.12) or ErrorException
end

@testset "no old names" begin
    for name in (:GPDecomposition, :X11Decomposition, :STLDecomposition,
        :AbstractDecomposition, :GPDecomp, :X11Decomp, :STLDecomp, :AbstractDecomp)
        @test !isdefined(GPDecompositions, name)
    end
end
