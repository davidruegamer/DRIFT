library(deepregression)
library(tidyverse)
library(ggplot2)
devtools::load_all("../neat/R")

set.seed(137)

### NAM block
feature_net <- function(x) x %>%
  layer_dense(units = 64 * 10, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 32 * 10, activation = "relu") %>%
  layer_dense(units = 1)

n <- 1000
x <- runif(n)
## @AT: Ideally we would like to go up to a factor of
## around 100-200 -- which are the cases in which B-splines fail
fac = 10

y <- sin(fac * x) + rnorm(n, sd = 0.4)
data <- data.frame(y = y, x = x)
x_test <- 1 + seq(0, 5, l=100)/10
y_test <- sin(fac * x_test) + rnorm(length(x_test), sd = 0.4)

### Models

### Structured regression
sr_fun <- function(kn, nam=FALSE){

  if(nam){
    meanform <- as.formula("~ 1 + snam(x)")
    opt <- optimizer_adam()
  }else{
    meanform <- as.formula(paste0("~ 1 + s(x, k = ", kn,
                                  ", df = ", kn, ")"))
    opt <- optimizer_adam(learning_rate = 0.1)
  }
  
  mod <- deepregression(y = y,
                        list_of_formulas = list(
                          meanform,
                          ~ 1
                        ),
                        data = data,
                        list_of_deep_models = list(snam = feature_net),
                        optimizer = opt
  )
  mod %>% fit(epochs = 2000L, verbose = FALSE,
              early_stopping = TRUE, patience = 50L)
  return(mod)

}

mod_a <- sr_fun(10)
y_fitted_a <- mod_a %>% predict(data)
y_test_fitted_a <- mod_a %>% predict(data.frame(x=x_test, y=y_test))

mod_b <- sr_fun(50)
y_fitted_b <- mod_b %>% predict(data)
y_test_fitted_b <- mod_b %>% predict(data.frame(x=x_test, y=y_test))

mod_c <- sr_fun(100)
y_fitted_c <- mod_c %>% predict(data)
y_test_fitted_c <- mod_c %>% predict(data.frame(x=x_test, y=y_test))

# mod_d <- sr_fun(1, TRUE)
# y_fitted_d <- mod_d %>% predict(data)
# y_test_fitted_d <- mod_d %>% predict(data.frame(x=x_test, y=y_test))


# Stuff for NEAT models

neat_mod <- neat(1, net_x_arch_trunk = feature_net, type="ls",
                 optimizer = optimizer_adam(learning_rate = 0.01))

neat_mod %>% fit(x = list(x, y), y, epochs = 5000L,
                 # callbacks = list(callback_early_stopping(monitor = "val_logLik",
                 #                                          patience = 500L,
                 #                                          restore_best_weights = TRUE)),
                 # validation_split = 0.1,
                 view_metrics = FALSE,
                 verbose = TRUE)

y_fitted_neat <- neat_mod %>% predict(list(x, y))

check_seq <- seq(-2.5, 2.5, l=100)

y_fitted_neat_all <- sapply(check_seq, function(add)
  neat_mod %>% predict(list(x, matrix(add, nrow = length(y)))) %>%
    as.matrix
)
matplot(t(y_fitted_neat_all), type="l")
colnames(y_fitted_neat_all) <- check_seq
rownames(y_fitted_neat_all) <- x
y_fitted_neat_all_med <- apply(y_fitted_neat_all, 1, function(x) check_seq[which.min(abs(x))])

check_seq_test <- seq(-2.5, 2.5, l=100)

y_test_fitted_neat <- sapply(check_seq_test, function(y)
  neat_mod %>% predict(list(x_test, matrix(y, nrow = length(y_test)))) %>%
    as.matrix
)
colnames(y_test_fitted_neat) <- check_seq_test
rownames(y_test_fitted_neat) <- x_test
y_test_fitted_neat_med <- apply(y_test_fitted_neat, 1, function(x) check_seq_test[which.min(abs(x))])

true_data <- data.frame(x = c(x, x_test),
                        y = c(y, y_test))
mod_data <- data.frame(x = rep(c(x, x_test), 4),
                       yhat = c(y_fitted_a, y_test_fitted_a,
                                y_fitted_b, y_test_fitted_b,
                                y_fitted_c, y_test_fitted_c,
                                # y_fitted_d, y_test_fitted_d
                                y_fitted_neat_all_med,
                                y_test_fitted_neat_med
                                ),
                       model = c(rep(c("GAM (10 basis functions)",
                                       "GAM (50 basis functions)",
                                       "GAM (100 basis functions)", 
                                       "NAM"),
                                     each = length(c(x, x_test)))),
                       part = c(rep(rep(c("in-dist.", "out-of-dist."), c(length(x), length(x_test))), 4))
)

mod_data$model <- factor(mod_data$model, levels = unique(mod_data$model))

ggplot() +
  geom_line(data = mod_data, aes(x = x, y = yhat, colour = model, linetype = part),
            linewidth = 1.2) + facet_wrap(~ model) +
  geom_point(data = true_data, aes(x = x, y = y), alpha = 0.5,
             shape = 4, size = 0.9) + theme_bw() +
  theme(text = element_text(size = 16),
        legend.title = element_blank(), legend.position = "bottom") +
  guides(color = "none") +
  xlab("x") + ylab("y") + xlim(0.5, 1.1)

ggsave(filename = "basis_fun.pdf", width = 10, height = 4)
