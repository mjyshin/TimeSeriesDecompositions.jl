derivative(f::AbstractGP) = DerivedGP((derivative, f), f.gpc)

const _DerivArgs = Tuple{typeof(derivative),AbstractGP}

function mean((_, _)::_DerivArgs, x::AbstractVector{<:Real})
    return zeros(length(x))
end

function cov((_, f)::_DerivArgs, x::AbstractVector{<:Real})
    return cov((derivative, f), x, x)
end

function var((_, f)::_DerivArgs, x::AbstractVector{<:Real})
    return diag(cov((derivative, f), x, x))
end

# Scalar kernel value at (t, t') via AbstractGP covariance
_kval(f::AbstractGP, t::Real, t′::Real) = cov(f, [t], [t′])[1, 1]

function cov((_, f)::_DerivArgs, x::AbstractVector{<:Real}, x′::AbstractVector{<:Real})
    # ∂²k/∂t∂t' via nested first-order ForwardDiff (avoids second-order dual issues)
    return [
        ForwardDiff.derivative(
            tn -> ForwardDiff.derivative(t′n -> _kval(f, tn, t′n), t′n), tn
        ) for tn in x, t′n in x′
    ]
end

function var((_, f)::_DerivArgs, x::AbstractVector{<:Real}, x′::AbstractVector{<:Real})
    return [
        ForwardDiff.derivative(
            tn -> ForwardDiff.derivative(t′n -> _kval(f, tn, t′n), t′n), tn
        ) for (tn, t′n) in zip(x, x′)
    ]
end

function cov(
    (_, f)::_DerivArgs, ::AbstractGP, x::AbstractVector{<:Real}, x′::AbstractVector{<:Real}
)
    # ∂k/∂t
    return [ForwardDiff.derivative(tn -> _kval(f, tn, t′n), tn) for tn in x, t′n in x′]
end

function cov(
    f′::AbstractGP, args::_DerivArgs, x::AbstractVector{<:Real}, x′::AbstractVector{<:Real}
)
    return collect(cov(args, f′, x′, x)')
end
