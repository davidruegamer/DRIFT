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

### Conditional
m <- PolrNN(ocmedv ~ 0 + nox, data = BostonHousing2,
            optimizer = optimizer_adam(learning_rate = 0.2, decay = 1e-4),
            latent_distr = "normal")
fit(m, epochs = 8e2, validation_split = NULL, batch_size = nrow(BostonHousing2))

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
  NEAT = -mod$test_step(list(list(tf$constant(X, dtype = "float32"), tf$constant(
    y, dtype = "float32")), y))$logLik$numpy()/nrow(X),
  TRAM = logLik(tm) / nrow(BostonHousing2),
  NONP = logLik(m) / nrow(BostonHousing2) # not same scale!
)

### Number of parameters
c(
  NEAT = mod$count_params(),
  TRAM = attr(logLik(tm), "df"),
  NONP = m$model$count_params()
)

# Plot --------------------------------------------------------------------

### TRAM
plot(tm, which = "baseline only")

### NONP
lines(levels(BostonHousing2$ocmedv)[-length(levels(BostonHousing2$ocmedv))],
      unlist(coef(m, which = "interacting")), type = "s", col = "red")

### NEAT
ys <- seq(min(y), max(y), length.out = 1e3)
Xs <- matrix(0, nrow = 1e3, ncol = 1)
lines(ys, c(predict(mod, list(Xs, matrix(ys)))), col = 3)

legend("topleft", c("Bernstein 25", "Nonparametric", "Location-scale NEAT"),
       col = 1:3, lwd = 2)

