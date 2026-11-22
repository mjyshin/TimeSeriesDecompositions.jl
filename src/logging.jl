## Opt-in suppression of @info/@warn messages ##

"""
    quietly(f; level=Logging.Error)

Run `f` with the current logger's minimum level raised to `level`, restoring it afterward
even if `f` throws.

# Example
```julia
quietly() do
    for y in series
        fit(GaussianProcess(), y, t; period=12)
    end
end
```

Pass `level=Logging.Warn` to hide `@info` but keep `@warn`.
"""
function quietly(f::Function; level::Logging.LogLevel = Logging.Error)
    return Logging.with_logger(f, MinLevelLogger(Logging.current_logger(), level))
end

"""
    @quietly expr
    @quietly level expr

Macro form of [`quietly`](@ref); `expr` can be a call, an assignment, a `for` loop or a
`begin...end` block. An assignment `@quietly x = f()` is `x = quietly(() -> f())`, so `x`
is assigned in the caller's scope.

# Example
```julia
d = @quietly fit(GaussianProcess(), y, t; period=12)   # hide everything below Error
@quietly Logging.Warn d = fit(X11(), y; period=12)     # hide @info, keep @warn
```

`fit` reports hyperparameters chosen from the data with `@info` (log group `:estimated`)
and emits `@warn` for fit diagnostics (log group `:diagnostics` for the smoothers).

!!! warning
    A `for` loop or `begin...end` block runs inside a closure: at top level (the REPL, a
    script or a notebook), assignments inside it do not reach global variables. Put
    `@quietly` on the assignment instead, `x = @quietly f()`.
"""
macro quietly(expr)
    return esc(_quietly(expr, Logging.Error))
end
macro quietly(level, expr)
    return esc(_quietly(expr, level))
end

# Wrap the right-hand side of an assignment (not of a method definition), else all of expr
function _quietly(expr, level)
    Meta.isexpr(expr, (:local, :global, :const), 1) &&
        return Expr(expr.head, _quietly(only(expr.args), level))
    _is_assignment(expr) &&
        return Expr(expr.head, expr.args[1], _quietly(expr.args[2], level))
    return :($quietly(() -> $expr; level = $level))
end

function _is_assignment(expr)
    return expr isa Expr && length(expr.args) == 2 && endswith(string(expr.head), "=") &&
           !Meta.isexpr(expr.args[1], (:call, :where))
end
