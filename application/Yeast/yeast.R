library(tidyverse)
library(ggplot2)
library(ggsci)
library(ggjoy)
library(devtools)
library(gamlss.mx)
if(!require(deepregression)){
  install_github("neural-structured-additive-learning/deepregression")
  library(deepregression)
}
if(!require(mixdistreg)){
  install_github("neural-structured-additive-learning/mixdistreg")
  library(mixdistreg)
}
load_all("../../neat/R")

set.seed(42)

### Pre-processing part
# data and comparison with https://academic.oup.com/bioinformatics/article/28/2/222/199665#2004541
data <- readRDS(file="yeastCC.RDS")
# extract data
exprs <- get("exprs", data@assayData)
# standardize
exprs <- scale(t(exprs))
exprs <- as.data.frame(exprs)
# only those with complete alpha time points
ind_alpha <- grepl("alpha", as.character(data$Timepoint))
exprs_alpha <- exprs[ind_alpha,]
exprs_alpha$classes <- data@phenoData@data$Phase[ind_alpha]
exprs_alpha$time <- gsub("alpha(.*)", "\\1", as.character(data$Timepoint[ind_alpha]))
exprs_alpha <- exprs_alpha[, !apply(exprs_alpha,2,function(x)any(is.na(x)))]
# get Spellman classification
# map to long format
data <- pivot_longer(exprs_alpha, cols=starts_with("Y"))
data$time <- as.numeric(data$time)
outcome <- data$value
data$name <- as.factor(data$name)

# Split dataset
set.seed(42)

idx_train = sample(seq_len(nrow(data)), nrow(data)*0.8, replace = FALSE)
train_data = data[idx_train,]
test_data = data[-idx_train,]


X_train = train_data %>% dplyr::select(time) %>% as.matrix 
y_train = train_data %>% dplyr::select(value) %>% as.matrix 
X_test = test_data %>% dplyr::select(time) %>% as.matrix 
y_test = test_data %>% dplyr::select(value) %>% as.matrix

### Model part
comps <- 6

mod <- sammer(y = y_train,
              list_of_formulas = list(~ 1 + s(time), 
                                      ~1 + s(time)), 
              formula_mixture = ~ 1,
              family = "normal", 
              data = data.frame(time = X_train),
              nr_comps = comps,
              tf_seed = 3
)

if(!file.exists(weights_mixtmod.hdf5)){
  
  # model fitting
  mod %>% fit(epochs = 2000, 
              validation_split = 0,
              batch_size = 256,
              view_metrics = FALSE, 
              verbose = F#T,
              # callbacks = 
              #   list(
              #     callback_early_stopping(patience = 500)#,
              #     # callback_reduce_lr_on_plateau(patience = 250)
              #   )
  )
  
  save_model_weights_hdf5(mod$model, filepath="weights_mixtmod.hdf5")

}else{
  
 mod$model$load_weights(filepath="weights_mixtmod.hdf5", by_name = FALSE)
  
}
  
mod %>% log_score(data = data.frame(time = X_test), 
                  this_y = y_test) %>% mean

plotdf <- expand.grid(time = seq(min(X_test), max(X_test), l=100),
                      y = seq(quantile(y_test, 0.01), quantile(y_test, 0.99), l=100))

dist <- mod %>% get_distribution(data = plotdf)
# post <- mod %>% get_pis(data = plotdf,
#                         this_y = plotdf$y,
#                         posterior = TRUE)
# NA_mask <- t(apply(post, 1, function(x){ 
#   res <- rep(NA, 6)
#   res[which.max(x)] <- 1
#   return(res)
#   }))
# qs <- lapply(seq(0.01, 0.99, l=100), function(q){ 
#   res <- cbind(as.matrix(tf$squeeze(dist$submodules[[1]]$quantile(value=q))) * NA_mask,
#         quantile = q, time = plotdf$time, y_test = y_test)
#   return(res) 
# }
# )
# qs <- do.call("rbind", qs) %>% as.data.frame
# colnames(qs)[9] <- "y_test"
# qs <- qs %>% pivot_longer(V1:V6)
# qs$name <- factor(qs$name, labels = paste0("Comp. ", 1:6))
# qs <- na.omit(qs)

pdfs <- as.matrix(tf$squeeze(dist$submodules[[1]]$prob(
  value = array(rep(plotdf$y, 6), dim = c(10000,1,6))))
)
dfpdf <- cbind(plotdf, dens_y = c(pdfs), comp = rep(1:6, each = 100^2))
dfpdf$comp <- factor(dfpdf$comp, labels = paste0("Mix. Comp. ", 1:6))

gg1 <- ggplot() + 
  geom_joy(data = dfpdf, 
           aes(height = dens_y*5, x = y + time/125 * 2, y = time, 
               group = time, fill = time), 
           stat="identity", alpha = 0.7, colour = rgb(0,0,0,0.4)) +
  theme_bw() + facet_wrap(~ comp, ncol=3) + xlab("") + ylab("Time") +
theme(panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      # panel.spacing = unit(-1.5, "lines"),
      strip.background = element_blank(),
      panel.border = element_blank(),
      text = element_text(size=14),
      legend.position = "none",
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      rect = element_rect(fill = "transparent"))

gg1

# ggplot(data = qs, aes(x = y_test, y = value)) + 
#   geom_point(aes(color = time), alpha = 0.2, shape = 3, size=2) + 
#   theme_bw() + theme(text = element_text(size = 16)) + 
#   xlab("True value") +
#   ylab("Transformation") +
#   guides(alpha="none") + 
#   facet_wrap(~ name, ncol = 3)
# 

ggsave(file = "yeast_mm.pdf", width = 5, height = 4)
ggsave(file = "yeast_mm_2.pdf", width = 8, height = 7)

### NEAT model

# Train the model
mod_neat_ls <- neat(1, type = "ls", 
                    net_y_size_trunk = nonneg_tanh_network(c(128, 128, 32)),
                    net_x_arch_trunk = relu_network(c(128, 128)),
                    optimizer = optimizer_adam())

if(file.exists("weights_neat.hdf5")){
  
  mod_neat_ls$load_weights(filepath="weights_neat.hdf5", by_name = FALSE)
  
}else{
  
  mod_neat_ls %>% fit(x = list(X_train,y_train), 
                      y = y_train, 
                      batch_size = 32L, epochs = 50L,
                      validation_split = 0.1, view_metrics = FALSE, 
                      callbacks = callback_early_stopping(
                        patience = 5, 
                        monitor="val_logLik",
                        restore_best_weights = TRUE)
  )
  
  save_model_weights_hdf5(mod_neat_ls, filepath="weights_neat.hdf5")

}
  
# Make predictions on test set
pred_neat_ls <- mod_neat_ls %>% predict(list(X_test,y_test))
- mod_neat_ls$evaluate(list(X_test, y_test), y_test) 

# plot per time (discrete)
# time_level = sort(as.vector(unique(X_train)))
df = data.frame(y_test = y_test, y_pred = pred_neat_ls, time = X_test)
# df$time_lev <- NA
# for(i in seq_along(time_level)) {
#   idx <- which(X_test[,1] == time_level[i])
#   df[idx, 'time_lev'] <- time_level[i]
# }
gg2 <- ggplot(data = df, aes(x = y_test, y = y_pred)) + 
  geom_point(aes(color = time), alpha = 0.2, size=2) + 
  theme_bw() + theme(text = element_text(size = 14)) + 
  xlab("True value") +
  ylab("Inverse flow values") +
  guides(alpha="none") + labs(color='Time') 

gg2

ggsave(file = "yeast_res.pdf", width = 5, height = 4)
ggsave(file = "yeast_res_2.pdf", width = 8, height = 7)

## pdf

plotdf <- expand.grid(time = seq(min(X_test), max(X_test), l=100),
                      y = seq(quantile(y_test, 0.001), quantile(y_test, 0.999), l=1000))

with(tf$GradientTape(persistent = TRUE) %as% tape, {
  
  x <- tf$convert_to_tensor(plotdf$time, dtype = tf$float32)
  y <- tf$convert_to_tensor(plotdf$y, dtype = tf$float32)

  tape$watch(x)
  tape$watch(y)
  
  # Feed forward
  h <- mod_neat_ls(list(x,y), training = FALSE)
  
  # Gradient and the corresponding loss function
  h_prime = tape$gradient(h, y)
  
})

pdf <- tfd_normal(0,1)$prob(h) %>% as.matrix + as.matrix(h_prime)
dfpdf_neat <- data.frame(time = plotdf$time, 
                         dens_y = pdf[,1], 
                         y = plotdf$y)

gg3 <- ggplot() + 
  geom_joy(data = dfpdf_neat %>% filter(time %in% quantile(dfpdf_neat$time, seq(0,1,l=10))), 
           aes(height = dens_y*5, x = y + time/125 * 5, y = time, 
               group = time, fill = time), 
           stat="identity", alpha = 0.7, colour = rgb(0,0,0,0.7)) +
  theme_bw() + ylab("Time") + xlab("") + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # panel.spacing = unit(-1.5, "lines"),
        strip.background = element_blank(),
        panel.border = element_blank(),
        text = element_text(size=14),
        legend.position = "none",
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        rect = element_rect(fill = "transparent"))

gg3

library(gridExtra)

lay <- rbind(
  c(1,2),
  c(1,3))

g <- grid.arrange(gg1, gg3, gg2, layout_matrix = lay)
ggsave(file = "yeast_matrix_fig.pdf", g, width = 8, height = 4)
