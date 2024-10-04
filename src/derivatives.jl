derivative(f::AbstractGP) = DerivedGP((derivative, f), f.gpc)

const deriv_args = Tuple{typeof(derivative), AbstractGP}

function mean((_, f)::deriv_args, x::AbstractVector{<:Real})
    return zeros(length(x))
    return ForwardDiff.derivative(α -> (mean(f, α * x)), 1.0)
end

function cov(args::deriv_args, x::AbstractVector{<:Real})
    return cov(args, x, x)
end
function var((_, f)::deriv_args, x::AbstractVector{<:Real})
    return var((derivative, f), x, x)
end

function cov((_, f)::deriv_args, x::AbstractVector{<:Real}, x′::AbstractVector{<:Real})
    fdm = central_fdm(5, 1)
    cols_of_C = map(x′) do x′ₙ
        col_elements = map(x) do xₙ
            FiniteDifferences.grad(
                fdm,
                xₙ -> FiniteDifferences.grad(
                    fdm,
                    x′ₙ -> cov(f, [xₙ], [x′ₙ]),
                    x′ₙ,
                ),
                xₙ
            )[1]
        end
    end
    return reduce(hcat, cols_of_C)
end
function var((_, f)::deriv_args, x::AbstractVector{<:Real}, x′::AbstractVector{<:Real})
    fdm = central_fdm(5, 1)
    elements = map(x, x′) do xₙ, x′ₙ
        FiniteDifferences.grad(
            fdm,
            xₙ -> FiniteDifferences.grad(
                fdm,
                x′ₙ -> cov(f, [xₙ], [x′ₙ]),
                x′ₙ,
            ),
            xₙ
        )[1]
    end
    return reduce(vcat, elements)
end

function cov(
    (_, f)::deriv_args,
    f′::AbstractGP,
    x::AbstractVector{<:Real},
    x′::AbstractVector{<:Real},
)
    fdm = central_fdm(5, 1)
    cols_of_C = map(x′) do x′ₙ
        col_elements = map(x) do xₙ
            FiniteDifferences.grad(
                fdm,
                xₙ -> cov(f, [xₙ], [x′ₙ]),
                xₙ,
            )[1]
        end
    end
    return reduce(hcat, cols_of_C)
end
function cov(
    f::AbstractGP,
    args::deriv_args,
    x::AbstractVector{<:Real},
    x′::AbstractVector{<:Real},
)
    return collect(cov(args, f, x′, x)')
end

fast_deriv(ρ_x̄::Function, t::Real) = gradient(mean ∘ ρ_x̄, t)[1]
fast_deriv(ρ_x̄::Function, t::AbstractVector{<:Real}) = fast_deriv.(Ref(ρ_x̄), t)
