library(dplyr)
library(deepregression)
library(deeptrafo)
devtools::load_all("~/NSL/deepoptim/")
optimizer_alig <- function(maxlr=NULL, mom=0.0, stab=1e-5) {
  python_path <- system.file("python", package = "deepoptim")
  opt <- reticulate::import_from_path("optimizers", path = python_path)
  return(opt$AliG(max_lr=maxlr, momentum=mom, eps=stab))
}

set.seed(32)

# Choose size
# size <- "small"
size <- "large"
save_model <- TRUE

# Read data
df <- read.csv("data/analysis_subset.csv")
df <- df %>% 
  rename(
    temp = average_temperature_celsius,
    humid = relative_humidity
  ) %>% 
  mutate(
    cases = log(new_confirmed * 1.0 + 1)
  ) %>% 
  filter(
    cases >=0
  )
# Train/Test
test <- df %>% filter(date > 700)
df <- df %>% filter(date <= 700)

if(size == "small")
{
  
  df <- df[sample(1:nrow(df), 5000),]
  bs <- 32L
  eps <- 2000L
  
}else{
  
  df <- df[sample(1:nrow(df), 1e6),]
  bs <- 128L
  eps <- 250L

}
ps <- 15L


# Define deep networks
deep_mod <- function(x) x %>% 
  layer_dense(units = 100, activation = "relu", use_bias = FALSE) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 100, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)

deep_mod_nam <- function(x) x %>% 
  layer_dense(units = 20, activation = "tanh", use_bias = FALSE) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 20, activation = "tanh") %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)

feature_net <- function(x) x %>% 
  layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
  layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

feature_net_bi <- function(x){ 
  
  base1 <- tf_stride_cols(x, 1) %>% 
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 32, activation = "relu") %>% 
    layer_dense(units = 5)
  
  base2 <- tf_stride_cols(x, 2) %>% 
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 32, activation = "relu") %>% 
    layer_dense(units = 5)
  
  tf_row_tensor(base1, base2) %>% 
    layer_dense(1)
  
}

# Define model formulas
form_structured <- ~ 1 + s(date) + s(population) + te(latitude, longitude) + 
  s(temp) + s(humid) 
form_nam <- ~ 1 + snam(date) + snam(population) + tenam(latitude, longitude) + 
  snam(temp) + snam(humid) 
form_sls <- cases ~ 1 + s(date) + s(population) + te(latitude, longitude) + 
  s(temp) + s(humid) 
form_neat <- cases  ~ 1 + snam(date) + snam(population) + tenam(latitude, longitude) + 
  snam(temp) + snam(humid) 

# Init and fit models
mod_str <- deepregression(
  y = df$cases,
  list_of_formulas = list(form_structured),
  data = df,
  family = "poisson",
  list_of_deep_models = NULL,
  orthog_options = orthog_control(orthogonalize = TRUE)
)

if(file.exists("weights_str.hdf5")){
  
  mod_str$model$load_weights(filepath="weights_str.hdf5", by_name = FALSE)
  
}else{
  
  hist_str <- mod_str %>% fit(epochs = eps, early_stopping = TRUE, 
                          patience = ps, batch_size = bs)
  
  if(save_model)
    save_model_weights_hdf5(mod_str$model, filepath="weights_str.hdf5")
  
}

# 584.7298
(metr_str <- Metrics::rmse(test$cases, mod_str %>% predict(test)))

plotdata_str <- plot(mod_str, only_data = TRUE)
saveRDS(plotdata_str, file = "plotdata_str.RDS")

rm(plotdata_str); gc(); gc()
rm(mod_str); gc(); gc()

### with NAM effects

mod_nam <- deepregression(
  y = df$cases,
  list_of_formulas = list(form_nam, ~1),
  data = df,
  list_of_deep_models = 
    list(
      snam = feature_net,
      tenam = feature_net_bi
    ),
  orthog_options = orthog_control(orthogonalize = FALSE)
)

if(file.exists("weights_nam.hdf5")){
  
  mod_nam$model$load_weights(filepath="weights_nam.hdf5", by_name = FALSE)
  
}else{
  
  hist_unc <- mod_nam %>% fit(epochs = eps, early_stopping = TRUE, 
                                    patience = ps, batch_size = bs)
  
  if(save_model)
    save_model_weights_hdf5(mod_nam$model, filepath="weights_nam.hdf5")
  
}

(metr_nam <- Metrics::rmse(test$cases, mod_nam %>% predict(test)))
# 589.0838

feature_eff_nets <- keras_model(inputs = mod_nam$model$inputs,
                                outputs = layer_concatenate(
                                  lapply(mod_nam$model$layers[[37]]$inbound_nodes[[1]]$inbound_layers,
                                         function(lay) lay$output)
                                )
)
nam_effects <- feature_eff_nets %>% predict(
  deepregression:::prepare_newdata(
    mod_nam$init_params$parsed_formulas_contents, df, 
    gamdata = mod_nam$init_params$gamdata$data_trafos
  )
)
colnames(nam_effects) <- trimws(strsplit(as.character(form_nam)[[2]], "+", fixed=TRUE)[[1]][c(2:6,1)])
saveRDS(nam_effects, file = "plotdata_nam.RDS")

rm(nam_effects); gc(); gc()
rm(mod_nam); gc(); gc()

### without dist assumption

mod_sls <- deeptrafo(form_sls, 
                     response_type = "count",
                     data = df,
                     list_of_deep_models = NULL,
                     orthog_options = orthog_control(orthogonalize = FALSE),
                     optimizer = optimizer_alig(0.5, mom = 0.9)
)

if(file.exists("weights_sls.hdf5")){
  
  mod_sls$model$load_weights(filepath="weights_sls.hdf5", by_name = FALSE)
  
}else{
  
  hist_sls <- mod_sls %>% fit(epochs = eps, early_stopping = TRUE, 
                              patience = ps, batch_size = bs*3#, 
                              # callbacks = callback_learning_rate_scheduler(
                              #   tf$keras$experimental$CosineDecayRestarts(.5, 5, t_mul = 2, m_mul = .8))
                              )
  
  if(save_model)
    save_model_weights_hdf5(mod_sls$model, filepath="weights_sls.hdf5")
  
}

(metr_sls <- Metrics::rmse(test$cases, mod_sls %>% predict(test)))
# 548.6764

plotdata_sls <- plot(mod_sls, only_data = TRUE)
saveRDS(plotdata_sls, file = "plotdata_sls.RDS")

rm(plotdata_sls); gc(); gc()
rm(mod_sls); gc(); gc()

### with NAM effects

neat <- deeptrafo(form_neat, 
                      response_type = "count",
                      data = df,
                      list_of_deep_models = list(
                        dnn = deep_mod_nam,
                        snam = feature_net,
                        tenam = feature_net_bi
                      ),
                      orthog_options = orthog_control(orthogonalize = FALSE)
)

if(file.exists("weights_neat.hdf5")){
  
  neat$model$load_weights(filepath="weights_neat.hdf5", by_name = FALSE)
  
}else{
  
  hist_nam <- neat %>% fit(epochs = eps, early_stopping = TRUE, 
                          patience = ps, batch_size = bs)
  
  if(save_model)
    save_model_weights_hdf5(neat$model, filepath="weights_neat.hdf5")
   
}

(metr_neat <- Metrics::rmse(test$cases, neat %>% predict(test)))
# 548.5575

# plot_y <- seq(min(df$cases), max(df$cases), l=1000)
# trafo_fun <- neat %>% predict(newdata = cbind(df[1:1000,-which(names(df)=="cases")], cases=plot_y), 
#                               type = "trafo")
# plot(trafo_fun ~ plot_y, type = "l")
# abline(0, 1, lty=2, col="red")

feature_eff_nets_neat <- keras_model(inputs = neat$model$inputs,
                                outputs = layer_concatenate(
                                  lapply(neat$model$layers[[51]]$inbound_nodes[[1]]$inbound_layers,
                                         function(lay) lay$output)
                                )
)
neatnam_effects <- feature_eff_nets_neat %>% predict(
  deepregression:::prepare_newdata(
    neat$init_params$parsed_formulas_contents, df, 
    gamdata = neat$init_params$gamdata$data_trafos
  )
)
colnames(neatnam_effects) <- trimws(strsplit(as.character(form_nam)[[2]], "+", fixed=TRUE)[[1]][c(2:6,1)])
saveRDS(neatnam_effects, file = "plotdata_neat.RDS")

rm(neatnam_effects); gc(); gc()
rm(neat); gc(); gc()

c(metr_str, metr_nam, metr_sls, metr_neat)