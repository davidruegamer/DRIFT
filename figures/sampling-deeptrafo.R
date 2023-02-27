# Structured additive sampling
# LK Feb 2023

set.seed(123)

# Dependencies ------------------------------------------------------------

library("deeptrafo")
library("tidyverse")
col3 <- colorspace::qualitative_hcl(3, alpha = 0.5)

# FUNs --------------------------------------------------------------------

### Baseline distributions
Y0_mixture <- \(ran = 10) {
  p <- \(y) 0.5 * pnorm(y, mean = -2) + 0.5 * pnorm(y, mean = 2)
  list(p = p,
       d = \(y) 0.5 * dnorm(y, mean = -2) + 0.5 * dnorm(y, mean = 2),
       q = Vectorize(\(p) {
         ret <- try(uniroot(\(y) p(y) - p, interval = c(-ran, ran))$root)
         if (inherits(ret, "try-error")) NA else ret
       }))
}

Y0_chisq <- \() {
  list(p = \(y) 0.5 * pchisq(y, df = 3) + 0.5 * pchisq(y, df = 6),
       d = \(y) 0.5 * dchisq(y, df = 3) + 0.5 * dchisq(y, df = 6),
       q = \(p) qchisq(p, df = 4))
}

### Shift and scale effects
ANP <- \() list(mu = \(x) exp(1 - exp(-x)) - 1, sigma = \(x) sqrt(exp(x)))

### Target distr
Z <- function() list(
  pZ = plogis,
  dZ = dlogis,
  qZ = qlogis,
  rZ = rlogis
)

get_Y <- function(Y0, Z, ANP) {
  ### Trafos
  g <- \(z, x = 0) Y0$q(Z$p(ANP$sigma(x) * z + ANP$mu(x)))
  h0 <- \(y) Z$q(Y0$p(y))
  hp <- \(y) (1 / Z$d(Z$q(Y0$p(y)))) * Y0$d(y)

  ### Full distr Y | X = x
  pY <- \(y, x) Z$p((1 / ANP$sigma(x)) * h0(y) - ANP$mu(x))
  dY <- \(y, x) Z$d((1 / ANP$sigma(x)) * h0(y) - ANP$mu(x)) * hp(y)

  list(p = pY, d = dY, g = g, sample = Vectorize(\(x) g(Z$rZ(1), x = x)))
}

Y <- get_Y(Y0_mixture(20), Z(), ANP())

# Sample ------------------------------------------------------------------

n <- 1e3
dd <- data.frame(X = runif(n, min = -3, max = 3))
dd$sX <- sqrt(exp(dd$X))
dd$Y <- Y$sample(dd$X)

# Fit ---------------------------------------------------------------------

dd$oY <- ordered(dd$Y)

bern <- ColrNN(Y | sX ~ 0 + s(X), data = dd)
nonp <- PolrNN(oY | sX ~ 0 + s(X), data = dd)

.to_gamma <- function(thetas) {
  gammas <- c(thetas[1L], log(exp(diff(thetas)) - 1))
  if (any(!is.finite(gammas))) {
    gammas[!is.finite(gammas)] <- 1e-20
  }
  gammas
}

w_init <- nonp$model$get_weights()
w_init[[1]][] <- thinit <- .to_gamma(qlogis(ecdf(dd$Y)(sort(dd$Y))))
w_init[[2]][] <- thinit / 2
nonp$model$set_weights(w_init)

fit(bern, batch_size = n, epochs = 1e4, optimizer = optimizer_adam(learning_rate = 1e-1), validation_split = FALSE)
fit(nonp, batch_size = n, epochs = 1e4, optimizer = optimizer_adam(learning_rate = 1e-1), validation_split = FALSE)

plot(c(predict(bern, type = "trafo")), c(predict(nonp, type = "trafo")))
abline(0, 1)

plot(bern)
plot(nonp)

nd <- data.frame(X = 0, sX = 1, Y = sort(dd$Y), oY = sort(dd$oY))
nd$bern <- predict(bern, newdata = nd, type = "cdf")
nd$nonp <- predict(nonp, newdata = nd, type = "cdf")
plot(nd$Y, nd$bern, type = "l", col = 2, main = "Baseline distribution")
lines(nd$Y, nd$nonp, type = "s", col = 3)
lines(nd$Y, Y$p(nd$Y, x = 0))
legend("topleft", c("Ground truth", "Bernstein", "Non-parametric"), col = 1:3, lwd = 1, bty = "n")
