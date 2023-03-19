source("main.R")

n <- 10
p <- 5
y <- sort(rnorm(11))
leny <- length(y)

X <- matrix(rnorm(n * p), ncol = p)[rep(1:n, leny),]
y <- matrix(rep(y, each = n))

mod <- neat(p, type = "ls")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 25L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(patience = 5, 
                                                restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")

mod <- neat(p, type = "tp")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 500L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(patience = 5, 
                                                restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")

mod <- neat(p, type = "inter")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 2500L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(patience = 250, 
                                                restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")

#### with NAMs

mod <- sneat(p, type = "ls")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 25L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(patience = 5, 
                                                restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")

mod <- sneat(p, type = "tp")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 500L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(patience = 5, 
                                                restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")

mod <- sneat(p, type = "inter")

mod %>% fit(x = list(X,y), y = y, batch_size = 32L, epochs = 500L,
            validation_split = 0.1, view_metrics = FALSE, 
            callbacks = callback_early_stopping(patience = 25, 
                                                restore_best_weights = TRUE))

pred <- mod %>% predict(list(X,y))

matplot(t(matrix(pred, ncol = 11)), type = "l")
