devtools::load_all("../")
library(deeptrafo)

#### Data simulation
n <- 10000
p <- 1
y <- rchisq(n, df = 4)
X <- matrix(rnorm(n * p), ncol = p)

# Plot options
par(mfrow=c(2,2))

#### NEAT
mod_neat <- neat(p, type = "ls", net_x_arch_trunk = function(x) tf$ones_like(x),
            optimizer = optimizer_adam(learning_rate = 0.0001), 
            net_y_size_trunk = nonneg_tanh_network(c(5,5)))

mod_neat %>% fit(x = list(X,y), y = y, batch_size = 400L, epochs = 500L,
            validation_split = 0.1, view_metrics = FALSE, 
            verbose = TRUE,
            callbacks = callback_early_stopping(
              patience = 100, 
              monitor="val_logLik",
              restore_best_weights = TRUE)
)

pred_neat <- mod_neat %>% predict(list(X,y))
plot(y ~ pred_neat)

qqnorm(y = pred_neat)
abline(0,1, col="red", lty=2)

#### Comparison deeptrafo
mod_dt <- deeptrafo(y ~ 1, data = data.frame(y=y), latent_distr = "normal",
                 optimizer = optimizer_adam(learning_rate = 0.0001))

mod_dt %>% fit(epochs=500L, early_stopping = TRUE, patience = 10L)

pred_dt <- predict(mod_dt)

plot(y ~ pred_dt)

qqnorm(pred_dt)
abline(0,1, col="red", lty=2)

