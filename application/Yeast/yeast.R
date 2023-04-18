library(tidyverse)
library(ggplot2)
library(ggsci)
library(devtools)
library(gamlss.mx)
load_all("~/NSL/deepregression")
load_all("~/NSL/mixdistreg")
load_all("~/NSL/safareg")

set.seed(42)

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

# get plot data
plotdata <- do.call("rbind", lapply(1:comps, function(i) cbind(as.data.frame(do.call("cbind", 
                                             plot(mod, which_dist = 1+i, only_data = T)[[1]][
                                               c("value", "partial_effect")]
                                             )), component = i))
)

plotdata2 <- do.call("rbind", lapply(1:comps, function(i) 
  cbind(as.data.frame(do.call("cbind", 
                              plot(mod, which_dist = 1+i, only_data = T,
                                   which_param = 2)[[1]][
                                c("value", "partial_effect")]
  )), component = i))
)

plotdata$sd <- plotdata2$V2

# calculate a posteriori probabilities
conv_fun <- function(x) as.matrix(tf$squeeze(x, axis=1L))
pis <- get_pis(mod)[1,]
dist <- mod %>% get_distribution()
apostprob_denpart <- conv_fun(dist$submodules[[1]]$prob(
  array(outcome, dim = c(NROW(outcome),1,1))))
apostprob_denpart <- aggregate(apostprob_denpart, data[,"name",drop=F], prod)
apostprob <- apostprob_denpart[,-1] * pis
maxapostprob <- apply(apostprob, 1, which.max)
data <- left_join(data, data.frame(component = maxapostprob, name=apostprob_denpart$name), by="name")

filter_mat <- t(sapply(data$component, function(x) 1:comps==x))

which_comps_have_data <- unique(as.character(data$component))

# plot
ggplot() + 
  geom_line(data=data %>% filter(component %in% which_comps_have_data), 
            aes(x=time, y=value, group=name), alpha=0.2) +
  geom_line(data=plotdata %>% filter(component %in% which_comps_have_data), 
            aes(x=value, y=V2), col="red") +
  geom_ribbon(data=plotdata %>% filter(component %in% which_comps_have_data),
              aes(x=value, ymin=V2-2*exp(sd),ymax=V2+2*exp(sd)), alpha=0.3, fill="red") + 
  facet_wrap(~component, ncol = 3) + theme_bw() + 
  xlab("Time") + ylab("Standardized expression level") + 
  theme(text = element_text(size = 16))

# ggsave("yeast_components.pdf", width=8, height=4)

