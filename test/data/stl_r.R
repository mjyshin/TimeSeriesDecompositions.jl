# Reference values for test/test_classical.jl from R's stl without robustness or jumps.
# Regenerate with: Rscript test/data/stl_r.R (from the repository root)
set.seed(11)
nextodd <- function(x) { n <- ceiling(x); if (n %% 2 == 0) n + 1 else n }
rows <- list()
for (case in list(c(12, 120, 7), c(7, 53, 7), c(4, 60, 11))) {
    p <- case[1]; N <- case[2]; ns <- case[3]
    t <- 1:N
    y <- 0.02 * t + sin(2 * pi * t / p) + 0.3 * sin(2 * pi * t / 37) + 0.3 * rnorm(N)
    nt <- nextodd(1.5 * p / (1 - 1.5 / ns))
    for (inner in 1:2) {
        m <- stl(ts(y, frequency = p), s.window = ns, s.degree = 1, t.window = nt,
            t.degree = 1, l.window = nextodd(p), l.degree = 1, inner = inner, outer = 0,
            robust = FALSE, s.jump = 1, t.jump = 1, l.jump = 1)
        rows[[length(rows) + 1]] <- c(p, N, ns, nt, inner, y, m$time.series[, "trend"],
            m$time.series[, "seasonal"])
    }
}
out <- file.path("test", "data", "stl_r.txt")
con <- file(out, "w")
for (r in rows) writeLines(paste(format(r, digits = 17), collapse = " "), con)
close(con)
