library(dplyr)
library(deepregression)
library(deeptrafo)
devtools::load_all("../../neat/")

set.seed(32)

save_model <- TRUE

# Read data
df <- read.csv("data/analysis_subset.csv")
df <- df %>% 
  rename(
    temp = average_temperature_celsius,
    humid = relative_humidity
  ) %>% 
  mutate(
    cases = new_confirmed,
    population = log(population),
    prevalence = log(prevalence*1000 + 1)
  ) %>% 
  filter(
    cases >=0
) %>% sample_n(500000)
# Train/Test
test <- df %>% filter(date > 700)
df <- df %>% filter(date <= 700)

# Network properties
bs <- 128L
eps <- 250L
ps <- 15L

# Define deep networks
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


# Init and fit models
mod_str <- deepregression(
  y = df$prevalence * 1.0,
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
(metr_str <- Metrics::rmse(test$prevalence, mod_str %>% predict(test)))

plotdata_str <- plot(mod_str, only_data = TRUE)
saveRDS(plotdata_str, file = "plotdata_str.RDS")

state_df <- map_data("state")
county_df <- map_data("county")

prplot <- county_df %>% select(lat, long) %>%
  rename(latitude = lat, longitude = long)

spatial_effect <- mod_str %>% get_partial_effect(names = "te(latitude, longitude)",
                                                 newdata = cbind(data.frame(date=0, population=0),
                                                                 prplot, 
                                                                 data.frame(temp=0, humid=0)))

saveRDS(spatial_effect, file = "plotspatial_str.RDS")

rm(plotdata_str); gc(); gc()
rm(mod_str); gc(); gc()

### with NAM effects

mod_nam <- deepregression(
  y = df$prevalence * 1.0,
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

(metr_nam <- Metrics::rmse(test$prevalence, mod_nam %>% predict(test)))
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



### NEAT

feature_net2 <- function(x) x %>% 
  layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 16, activation = "relu") 

feature_net_bi2 <- function(x){ 
  
  base1 <- tf_stride_cols(x, 1) %>% 
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 32, activation = "relu") %>% 
    layer_dense(units = 16, activation = "relu") %>% 
    layer_dense(units = 5)
  
  base2 <- tf_stride_cols(x, 2) %>% 
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 32, activation = "relu") %>% 
    layer_dense(units = 16, activation = "relu") %>% 
    layer_dense(units = 5)
  
  tf_row_tensor(base1, base2)
  
}

structured_nam_network <- function(input){
  
  list_of_features <- tf$split(input, 
                               num_or_size_splits = c(1L,1L,1L,2L,1L,1L), 
                               axis=1L)
  funs <- list(
    function(x) x,
    feature_net2, feature_net2,
    feature_net_bi2, 
    feature_net2, feature_net2
  )
  
  list_of_outputs <- lapply(1:length(list_of_features),
                            function(i) funs[[i]](list_of_features[[i]]))
  
  return(layer_concatenate(list_of_outputs))
  
}

neat_mod <- neat(dim_features = 7, type = "ls",
                 net_y_size_trunk = nonneg_tanh_network(c(10, 10)),
                 net_x_arch_trunk = structured_nam_network,
                 optimizer = optimizer_adam(learning_rate = 0.0001)
)

if(file.exists("weights_neat.hdf5")){
  
  neat_mod$load_weights(filepath="weights_neat.hdf5", by_name = FALSE)
  
}else{
  
  hist_neat <- neat_mod %>% fit(x = list(cbind(1, as.matrix(
    df[,c("date", "population",
          "latitude", "longitude",
          "temp", "humid")])),
    matrix(df$prevalence)), 
    y = matrix(df$prevalence),
    epochs = eps, 
    callbacks = callback_early_stopping(
      patience = ps*10, restore_best_weights = TRUE,
      monitor = "val_logLik"
    ),
    view_metrics = FALSE,
    validation_split = 0.1,
    batch_size = bs)
  
  if(save_model)
    save_model_weights_hdf5(neat_mod, filepath="weights_neat.hdf5")
   
}

pred_mod_neat <- keras_model(inputs = neat_mod$inputs,
                             outputs = neat_mod$layers[[prelast]]$output
)

pred_neat <- pred_mod_neat %>% predict(
  list(cbind(1,as.matrix(test[,c("date", "population",
                               "latitude", "longitude",
                               "temp", "humid")])),
       matrix(test$prevalence))
)
(metr_neat <- Metrics::rmse(test$prevalence, pred_neat))

# plot(pred_neat ~ log(test$prevalence+1))

prelast <- length(neat_mod$layers)-1

get_partial_effects <- function(){
  
  size_uni <- 16L
  size_bi <- 25L
  sizes <- c(1L, size_uni, size_uni, size_bi, size_uni, size_uni)
  
  bases <- tf$split(neat_mod$layers[[prelast]]$inbound_nodes[[1]]$inbound_layers$output,
                    sizes, axis = 1L)
  ws <- tf$split(neat_mod$layers[[prelast]]$weights[[1]], 
                 sizes, axis = 0L)
  
  layer_concatenate(lapply(1:6, function(i) tf$matmul(bases[[i]], ws[[i]])))
  
}

feature_eff_nets_neat <- keras_model(inputs = neat_mod$inputs,
                                outputs = get_partial_effects()
                                )

neatnam_effects <- feature_eff_nets_neat %>% predict(
  list(cbind(1,as.matrix(df[,c("date", "population",
                       "latitude", "longitude",
                       "temp", "humid")])),
       matrix(df$prevalence))
)
colnames(neatnam_effects) <- trimws(strsplit(as.character(form_nam)[[2]], "+", fixed=TRUE)[[1]][c(1:6)])
saveRDS(neatnam_effects, file = "plotdata_neat.RDS")

state_df <- map_data("state")
county_df <- map_data("county")

prplot <- county_df %>% select(lat, long) %>%
  rename(latitude = lat, longitude = long)

spatial_effect <- feature_eff_nets_neat %>% predict(
  list(as.matrix(cbind(1, 0, 0, prplot, 0, 0)), matrix(rep(0, nrow(prplot))))
)

saveRDS(spatial_effect, file = "plotspatial_neat.RDS")

rm(neatnam_effects); gc(); gc()
rm(neat_mod); gc(); gc()

# compare metrics
c(metr_str, metr_nam, metr_neat)
