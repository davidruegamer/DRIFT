devtools::load_all("../")
library(deeptrafo)

#### Data simulation
n <- 100
p <- 5
y <- sort(rnorm(100))
leny <- length(y)

X <- matrix(rnorm(n * p), ncol = p)[rep(1:n, leny),]
y <- matrix(rep(y, each = n)) - X%*%c(-2:2)

#### Model types comparison
mod <- neat(p, type = "ls", net_x_arch_trunk = function(x) x,
            optimizer = optimizer_adam())

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 500L,
            validation_split = 0.1, view_metrics = FALSE, 
            verbose = TRUE,
            callbacks = callback_early_stopping(
              patience = 20, 
              monitor="val_logLik",
              restore_best_weights = TRUE)
)

pred <- mod %>% predict(list(X,y))

(logLik <- - mod$evaluate(list(as.matrix(X),
                               matrix(y)),
                          matrix(y))/nrow(X))


par(mfrow=c(1,2))
matplot(t(matrix(pred, ncol = leny)), type = "l")

data <- cbind(data.frame(y=y), X)
colnames(data)[-1] <- paste0("x", 1:5)

mod_dt <- deeptrafo(formula = y ~ x1 + x2 + x3 + x4 + x5, 
                    data = data)
mod_dt %>% fit(epochs = 1000L, verbose = TRUE,
            early_stopping = TRUE, patience = 50)

pred_dt <- predict(mod_dt, type = "trafo")
matplot(t(matrix(pred_dt, ncol = leny)), type = "l")

plot(pred ~ pred_dt)

####### INTER


mod <- neat(p, type = "inter")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 1000L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(patience = 250, 
                                                monitor="val_logLik",
                                                restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

(logLik <- - mod$evaluate(list(as.matrix(X),
                               matrix(y)),
                          matrix(y))/nrow(X))

matplot(t(matrix(pred, ncol = 11)), type = "l")