devtools::load_all("../")

#### Data simulation
n <- 10
p <- 5
y <- sort(rnorm(11))
leny <- length(y)

X <- matrix(rnorm(n * p), ncol = p)[rep(1:n, leny),]
y <- matrix(rep(y, each = n))

#### Model types comparison
mod <- neat(p, type = "ls")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 25L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(
              patience = 5, 
              monitor="val_logLik",
              restore_best_weights = TRUE)
            )

pred <- mod %>% predict(list(X,y))

(logLik <- - mod$evaluate(list(as.matrix(X),
                          matrix(y)),
                     matrix(y))/nrow(X))


matplot(t(matrix(pred, ncol = 11)), type = "l")

mod <- neat(p, type = "tp")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 500L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(patience = 5, 
                                                monitor="val_logLik",
                                                restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

## ll
(logLik <- - mod$evaluate(list(as.matrix(X),
                               matrix(y)),
                          matrix(y))/nrow(X))

matplot(t(matrix(pred, ncol = 11)), type = "l")

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

#### with NAMs

mod <- sneat(p, type = "ls")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, 
            epochs = 250L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(
              patience = 5, 
              monitor="val_logLik",
              restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")

mod <- sneat(p, type = "tp")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, 
            epochs = 500L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(
              patience = 5, 
              monitor="val_logLik",
              restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")

mod <- sneat(p, type = "inter")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 500L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(patience = 25, monitor="val_logLik",
                                                restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")

#### with semi-structured NAMs

mod <- sesneat(p, type = "ls")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, 
            epochs = 250L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(
              patience = 5, 
              monitor="val_logLik",
              restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")

mod <- sesneat(p, type = "tp")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, 
            epochs = 500L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(
              patience = 5, 
              monitor="val_logLik",
              restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")

mod <- sesneat(p, type = "inter")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 500L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(patience = 25, monitor="val_logLik",
                                                restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")
