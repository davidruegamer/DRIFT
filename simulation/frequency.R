library(deepregression)
library(tidyverse)
library(ggplot2)
devtools::load_all("../../neat/R")

set.seed(137)

### NAM block
feature_net <- function(x) x %>% 
  layer_dense(units = 64 * 10, activation = "relu", use_bias = FALSE) %>% 
  layer_dense(units = 32 * 10, activation = "relu") %>% 
  layer_dense(units = 1)

n <- 1000
x <- runif(n)
fac = 200
  
y <- sin(fac * x) + rnorm(n, sd = 0.4)
data <- data.frame(y = y, x = x)
x_test <- 1 + seq(0, 5, l=100)/10
y_test <- sin(fac * x_test) + rnorm(length(x_test), sd = 0.4)
    
### Models

### Structured regression
sr_fun <- function(kn){
  
  mod <- deepregression(y = y,
                        list_of_formulas = list(
                          as.formula(paste0("~ 1 + s(x, k = ", kn, 
                                            ", df = ", kn, ")")),
                          ~ 1
                        ),
                        data = data,
                        optimizer = optimizer_adam(learning_rate = 0.1)
  )
  mod %>% fit(epochs = 2000L, verbose = FALSE,
              early_stopping = TRUE, patience = 50L)
  return(mod)
  
}

mod_10 <- sr_fun(10)
y_fitted_10 <- mod_10 %>% predict(data)
y_test_fitted_10 <- mod_10 %>% predict(data.frame(x=x_test, y=y_test))

mod_20 <- sr_fun(20)
y_fitted_20 <- mod_20 %>% predict(data)
y_test_fitted_20 <- mod_20 %>% predict(data.frame(x=x_test, y=y_test))

mod_50 <- sr_fun(50)
y_fitted_50 <- mod_50 %>% predict(data)
y_test_fitted_50 <- mod_50 %>% predict(data.frame(x=x_test, y=y_test))
    
neat_mod <- neat(1, net_x_arch_trunk = feature_net)
    
neat_mod %>% fit(x = list(x, y), y, epochs = 2000L, 
                 callbacks = list(callback_early_stopping(monitor = "val_logLik",
                                                          patience = 50L, 
                                                          restore_best_weights = TRUE)),
                 validation_split = 0.1,
                 verbose = FALSE)

y_fitted_neat <- neat_mod %>% predict(list(x, y))
y_test_fitted_neat <- sapply(seq(-2.5, 2.5, l=100), function(y)
  pred_cdf <- tfd_normal(0,1) %>% 
    tfd_cdf(neat_mod %>% predict(list(x_test, rep(y, length(x_test))))) %>% 
    as.matrix
)
colnames(y_test_fitted_neat) <- #paste0("y = ", 
  seq(-2.5, 2.5, l=100)#)
rownames(y_test_fitted_neat) <- x_test
y_test_fitted_neat_med <- apply(y_test_fitted_neat, 1, function(x) seq(-2.5, 2.5, l=100)[which.min(abs(x-0.5))])

true_data <- data.frame(x = c(x, x_test), 
                        y = c(y, y_test))
mod_data <- data.frame(x = rep(c(x, x_test), 4),
                       yhat = c(y_fitted_10, y_test_fitted_10,
                                y_fitted_20, y_test_fitted_20,
                                y_fitted_50, y_test_fitted_50,
                                y_fitted_neat, y_test_fitted_neat_med),
                       model = c(rep(c("GAM (10)", "GAM (20)", "GAM (50)", "NEAT"), 
                                     each = length(c(x, x_test)))),
                       part = c(rep(rep(c("in-dist.", "out-of-dist."), c(length(x), length(x_test))), 4))
)
  
ggplot() + 
  geom_line(data = mod_data, aes(x = x, y = yhat, colour = model, linetype = part),
            size = 1.2) + facet_wrap(~ model) + 
  geom_point(data = true_data, aes(x = x, y = y), alpha = 0.5,
             shape = 4) + theme_bw() + 
  theme(text = element_text(size = 16),
        legend.title = element_blank()) + 
  xlab("x") + ylab("y") + xlim(0.5, 1.1)
ggsave(filename = "basis_fun.pdf", width = 5, height = 4)
