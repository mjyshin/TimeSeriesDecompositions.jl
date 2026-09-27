# Reference values for test/test_filters.jl from R's loess with exact computation.
# Regenerate with: Rscript test/data/loess_r.R (from the repository root)
set.seed(7)
t <- sort(runif(50, 0, 20))
y <- sin(t) + 0.2 * rnorm(50)
w <- runif(50, 0.2, 2)
tn <- seq(min(t), max(t), length.out = 17)
rows <- list()
for (weighted in 0:1) for (degree in 0:2) for (span in c(0.2, 0.5, 0.9, 1.5)) {
    weights <- if (weighted == 1) w else rep(1, 50)
    m <- loess(
        y ~ t, weights = weights, span = span, degree = degree, surface = "direct",
        family = "gaussian"
    )
    rows[[length(rows) + 1]] <- c(
        weighted, degree, span, m$trace.hat, predict(m, data.frame(t = tn)), fitted(m)
    )
}
out <- file.path("test", "data")
write.table(cbind(t, y, w), file.path(out, "loess_r_data.txt"), row.names = FALSE, col.names = FALSE)
write.table(do.call(rbind, rows), file.path(out, "loess_r_fits.txt"), row.names = FALSE, col.names = FALSE)
