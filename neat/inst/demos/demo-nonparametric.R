# Demo continuous response -- nonparametric estimator
# Lucas Kook
# Feb 2022

set.seed(1234)

# Deps --------------------------------------------------------------------

library("tram")
library("deeptrafo")
devtools::load_all(".")

# Data --------------------------------------------------------------------

data("BostonHousing2", package = "mlbench")
BostonHousing2 <- BostonHousing2 %>%
  dplyr::mutate(cmedv = c(scale(cmedv)), ocmedv = ordered(cmedv), nox = c(scale(nox)))

X <- model.matrix(~ nox, data = BostonHousing2)[, -1, drop = FALSE]
y <- matrix(BostonHousing2$cmedv, ncol = 1)

# Model -------------------------------------------------------------------

tm <- BoxCox(cmedv ~ nox, data = BostonHousing2, order = 25, support = range(y))
tmls <- BoxCox(cmedv ~ nox | nox, data = BostonHousing2, order = 25, support = range(y))
tmtp <- BoxCox(cmedv | nox ~ 1, data = BostonHousing2, order = 25, support = range(y))

### Nonparametric
m <- PolrNN(ocmedv ~ 0 + nox, data = BostonHousing2,
            optimizer = optimizer_adam(learning_rate = 0.2, decay = 1e-4),
            latent_distr = "normal")
fit(m, epochs = 8e2, validation_split = NULL, batch_size = nrow(BostonHousing2))

### Nonparametric TP
# mtp <- PolrNN(ocmedv | nox ~ 1, data = BostonHousing2,
#             optimizer = optimizer_adam(learning_rate = 1e-3, decay = 1e-4),
#             latent_distr = "normal")
# tmp <- get_weights(mtp$model)
# tmp[[1]][] <- tmp[[2]][] <- get_weights(m$model)[[1]]
# tmp[[3]][] <- 0
# set_weights(mtp$model, tmp)
# fit(mtp, epochs = 3e3, validation_split = NULL, batch_size = 16)

### NEAT location scale
mod <- neat(p <- ncol(X), type = "ls", optimizer = optimizer_adam(learning_rate = 1e-2, decay = 1e-3))
mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 1e3,
            validation_split = 0.1, view_metrics = FALSE,
            callbacks = callback_early_stopping(patience = 100,
                                                monitor="val_logLik",
                                                restore_best_weights = TRUE))

# Eval --------------------------------------------------------------------

### In-sample logLik
c(
  LOC = logLik(tm) / nrow(BostonHousing2),
  LOCSCA = logLik(tmls) / nrow(BostonHousing2),
  TP = logLik(tmtp) / nrow(BostonHousing2),
  NONP = logLik(m) / nrow(BostonHousing2), # not same scale!
  # NONPTP = logLik(mtp) / nrow(BostonHousing2), # not same scale!
  NEAT = -mod$test_step(list(list(tf$constant(X, dtype = "float32"), tf$constant(
    y, dtype = "float32")), y))$logLik$numpy()/nrow(X)
)

### Number of parameters
c(
  LOC = attr(logLik(tm), "df"),
  LOCSCA = attr(logLik(tmls), "df"),
  TP = attr(logLik(tmtp), "df"),
  NONP = m$model$count_params(),
  # NONPTP = mtp$model$count_params(),
  NEAT = mod$count_params()
)

# Plot --------------------------------------------------------------------

### TRAM
plot(tm, which = "distribution", type = "trafo", newdata = data.frame(nox = 0))
plot(tmls, which = "distribution", type = "trafo", newdata = data.frame(nox = 0), add = TRUE, lty = 2)
plot(tmtp, which = "distribution", type = "trafo", newdata = data.frame(nox = 0), add = TRUE, lty = 3)

### NONP
lines(levels(BostonHousing2$ocmedv)[-length(levels(BostonHousing2$ocmedv))],
      unlist(coef(m, which = "interacting")), type = "s", col = "red")

### NONP TP
# plot(mtp, type = "trafo", newdata = data.frame(nox = c(0, 0)))

### NEAT
ys <- seq(min(y), max(y), length.out = 1e3)
Xs <- matrix(0, nrow = 1e3, ncol = 1)
lines(ys, c(predict(mod, list(Xs, matrix(ys)))), col = 3)

legend("topleft", c("Bernstein 25", "Bernstein 25 LS", "Bernstein 25 TP",
                    "Nonparametric", "Location-scale NEAT"),
       col = c(1, 1, 1:3), lwd = 2, lty = c(1, 2, 3, 1, 1))

