function ℒ(θ, 𝓉, 𝓎)
    𝒢𝒫 = build_gp(θ)
    return -logpdf(𝒢𝒫(𝓉, 0.0), 𝓎)    # NLL with zero data noise
end

default_optimizer = LBFGS(;
    alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),
)

function optimize_loss(loss, θ0; optimizer=default_optimizer, maxiter=1_000)
    options = Optim.Options(; iterations=maxiter, show_trace=false)

    θ0_flat, unflatten = ParameterHandling.value_flatten(θ0)
    loss_packed = loss ∘ unflatten

    # https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/#avoid-repeating-computations
    function fg!(F, G, x)
        if F !== nothing && G !== nothing
            val, grad = Zygote.withgradient(loss_packed, x)
            G .= only(grad)
            return val
        elseif G !== nothing
            grad = Zygote.gradient(loss_packed, x)
            G .= only(grad)
            return nothing
        elseif F !== nothing
            return loss_packed(x)
        end
    end

    result = optimize(Optim.only_fg!(fg!), θ0_flat, optimizer, options; inplace=false)

    return unflatten(result.minimizer)
end
