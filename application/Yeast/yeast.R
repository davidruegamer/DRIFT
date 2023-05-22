library(tidyverse)
library(ggplot2)
library(ggsci)
library(devtools)
library(gamlss.mx)
if(require(deepregression)){
  install_github("neural-structured-additive-learning/deepregression")
}
if(require(mixdistreg)){
  install_github("neural-structured-additive-learning/mixdistreg")
}

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

### Model part
comps <- 6

mod <- sammer(y = outcome,
              list_of_formulas = list(~ 1 + s(time), 
                                      ~1 + s(time)), 
              formula_mixture = ~ 1,
              family = "normal", 
              data = data,
              nr_comps = comps,
              tf_seed = 3
)

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


mod %>% log_score(data = data, this_y = matrix(outcome)) %>% mean
