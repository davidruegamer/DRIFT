# Structured additive sampling
# LK Feb 2023

# Dependencies ------------------------------------------------------------

library("tidyverse")
col3 <- colorspace::qualitative_hcl(3, alpha = 0.5)

# FUNs --------------------------------------------------------------------

pY0 <- \(y) 0.5 * pnorm(y, mean = -2) + 0.5 * pnorm(y, mean = 2)
# 0.5 * pchisq(y, df = 3) + 0.5 * pchisq(y, df = 6)
dY0 <- \(y) 0.5 * dnorm(y, mean = -2) + 0.5 * dnorm(y, mean = 2)
# 0.5 * dchisq(y, df = 3) + 0.5 * dchisq(y, df = 6)
qY0 <- Vectorize(\(p) {
  ret <- try(uniroot(\(y) pY0(y) - p, interval = c(-10, 10))$root)
  if (inherits(ret, "try-error")) NA else ret
}) # qchisq(p, df = 4)

ranY <- qY0(c(0.001, 0.999))

pZ <- plogis
dZ <- dlogis
qZ <- qlogis
rZ <- rlogis

sigma <- \(x) sqrt(exp(x))
beta <- \(x) exp(1 - exp(-x)) - 1

g <- \(z, x = 0) qY0(pZ(sigma(x) * z + beta(x)))
h0 <- \(y) qZ(pY0(y))
hp <- \(y) (1 / dZ(qZ(pY0(y)))) * dY0(y)

pY <- \(y, x) pZ((1 / sigma(x)) * h0(y) - beta(x))
dY <- \(y, x) dZ((1 / sigma(x)) * h0(y) - beta(x)) * hp(y)

# Plot --------------------------------------------------------------------

z <- seq(-4, 4, length.out = 1e3)
y <- seq(ranY[1], ranY[2], length.out = 1e3)
dz <- dZ(z)
x <- c(-0.5, 0, 0.5)
Y <- sapply(x, \(tx) g(z, tx))
dy <- sapply(x, \(tx) dY(y, tx))

opar <- par(no.readonly = TRUE)
pdf("sampling.pdf", width = 4, height = 4)
par(mar = rep(0.1, 4))
layout(rbind(c(1, 2), c(3, 4)), widths = c(2, 1, 2, 1), heights = c(1, 2, 1, 2))
plot(z, dz, type = "l", axes = FALSE)
# box()
plot.new()
legend("topleft", legend = x, col = col3, bty = "n", lwd = 1, title = "x", cex = 1.5)
matplot(z, Y, type = "l", axes = FALSE, lty = 1, col = col3)
text(0, par("usr")[4] - 0.75, expression(epsilon), cex = 1.5)
text(par("usr")[2] - 0.75, 0, "Y | X = x", cex = 1.5, srt = -90, xpd = TRUE)
text(0, par("usr")[1] + 0.75, parse(text = "phi(epsilon, x)"), cex = 1.5)
box()
matplot(dy, y, type = "l", axes = FALSE, lty = 1, col = col3)
# box()
dev.off()

par(opar)
pdf("samples.pdf", width = 3.5, height = 3.5)
set.seed(8)
n <- 1e3
smpl <- sapply(x, \(tx) g(rlogis(n), x = tx))
# boxplot(smpl, col = col3, axes = FALSE, xlab = "x", ylab = "y", pch = 20, outcol = col3)
beeswarm::beeswarm(as.data.frame(smpl), col = col3, axes = FALSE, xlab = "x",
                   ylab = "y", pch = 20, cex = 0.25, method = "hex")
box()
axis(1, at = 1:3, labels = x)
axis(2, las = 1)
dev.off()
