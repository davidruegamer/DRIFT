############################# Loading libraries #############################
library(deeptrafo)
library(Metrics)

############################# Architectures #############################
get_deep_mod <- function(architecture = c("a", "b", "c", "d")){
  
  architecture <- match.arg(architecture)
  deep_mod <- switch(architecture,
                     a = function(x) x %>% 
                       layer_dense(units = 200, activation = "relu", use_bias = FALSE) %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 1, activation = "linear"),
                     b = function(x) x %>% 
                       layer_dense(units = 200, activation = "relu", use_bias = FALSE) %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 200, activation = "relu") %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 1, activation = "linear"),
                     c = function(x) x %>% 
                       layer_dense(units = 20, activation = "relu", use_bias = FALSE) %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 1, activation = "linear"),
                     d = function(x) x %>% 
                       layer_dense(units = 20, activation = "relu", use_bias = FALSE) %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 20, activation = "relu") %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 1, activation = "linear")
  )
  
  return(deep_mod)
}

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

###################### Generic Normal (Deep) Regression #######################
dr <- function(formla, trainX, trainY, testX, testY,
               deep_mod_list = NULL,
               additional_processors = NULL,
               maxEpochs = 1000,
               patience = 50, 
               optimizer = optimizer_adam(),
               verbose = FALSE,
               oz_option = orthog_control(),
               ...
               ){
  
  family <- "normal"
    
  if(length(unique(trainY))<=5) family = "multinomial"

  args <- list(y = trainY, 
               list_of_formulas = list(as.formula(formla), 
                                       as.formula(formla)),
               list_of_deep_models = deep_mod_list,
               additional_processors = additional_processors,
               data = trainX, 
               family = family,
               optimizer = optimizer,
               orthog_options = oz_option)
  
  ellips <- list(...)
  args <- c(args, ellips)

  mod <- do.call("deepregression", args)
  
  mod %>% fit(epochs = maxEpochs, early_stopping = TRUE, 
              patience = patience, verbose = verbose)
  
  # res <- mod %>% predict(testX)
  ll <- log_score(mod, data = testX, this_y = testY)

  return(ll)
  
}

#################### Generic (Deep) Transformation Model ######################

tm <- function(formula, trainX, trainY, testX, testY,
               deep_mod_list = NULL,
               additional_processors = NULL,
               maxEpochs = 1000,
               patience = 50, 
               optimizer = optimizer_adam(),
               verbose = FALSE,
               oz_option = orthog_control(),
               ...
)
{
  
  # assuming y is response in formula
  data = cbind(trainX, y = trainY)
  
  family <- "normal"
  
  args <- list(formula = formula,
               data = data,
               latent_distr = family,
               list_of_deep_models = deep_mod_list,
               additional_processors = additional_processors,
               data = trainX, 
               family = family,
               optimizer = optimizer,
               orthog_options = oz_option)
  
  ellips <- list(...)
  args <- c(args, ellips)
  
  mod <- do.call("deeptrafo", args)
  
  mod %>% fit(epochs = maxEpochs, early_stopping = TRUE, 
              patience = patience, verbose = verbose)
  
  # res <- mod %>% predict(testX)
  ll <- logLik(mod, newdata = cbind(testX, y = testY))
  
  return(ll)
  
}

############################# DDR #############################
ddr <- function(trainX, trainY, testX, testY,
                architecture,
                ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  Vs <- colnames(trainX)
  
  form <- paste0("~ 1", 
                 " + deep_mod(",
                 paste(Vs, collapse=", "), ")")
  
  res <- dr(form, deep_mod_list = list(deep_mod = deep_mod),
             trainX = trainX, trainY = trainY, testX = testX,
             ...)
  
  return(res)
  
  
}
############################# SADR #############################
sadr <- function(trainX, trainY, testX, testY,
                 ...){
  
  Vte <- c("latitude", "longitude")
  Vs <- setdiff(colnames(trainX), Vte)
  
  form <- paste0("~ 1 + ",
                 "te(", Vte[1], ", ", Vte[2], ") + ",
                 paste(paste0("s(", Vs, ")"), collapse=" + "))
  
  res <- dr(form, trainX, trainY, testX, ...)
  
  return(res)
  
  
}

############################# SSDR #############################
ssdr <- function(trainX, trainY, testX, testY,
                 architecture,
                 ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  Vte <- c("latitude", "longitude")
  Vs <- setdiff(colnames(trainX), Vte)
  
  form <- paste0("~ 1 + ",
                 "te(", Vte[1], ", ", Vte[2], ") + ",
                 paste(paste0("s(", Vs, ")"), collapse=" + "),
                 " + deep_mod(",
                 paste(Vs, collapse=", "), ")")
  
  res <- dr(form, deep_mod_list = list(deep_mod = deep_mod),
             trainX, trainY, testX, 
             oz_option = orthog_control(orthogonalize = FALSE),
             ...)
  
  return(res)
  
  
}

############################# NAMDR #############################
namdr <- function(trainX, trainY, testX, testY,
                  architecture,
                  ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  Vte <- c("latitude", "longitude")
  Vs <- setdiff(colnames(trainX), Vte)
  
  form <- paste0("~ 1 + ",
                 "tenam(", Vte[1], ", ", Vte[2], ") + ",
                 paste(paste0("snam(", Vs, ")"), collapse=" + "))
  
  res <- dr(form, 
             trainX, trainY, testX, 
             oz_option = orthog_control(orthogonalize = FALSE),
             list_of_deep_models = list(
               deep_mod = deep_mod,
               snam = feature_net,
               tenam = feature_net_bi
             ),
             ...)
  
  return(res)
  
  
}

############################# SNAMDR #############################
snamdr <- function(trainX, trainY, testX, testY,
                   architecture,
                   ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  Vte <- c("latitude", "longitude")
  Vs <- setdiff(colnames(trainX), Vte)
  
  form <- paste0("~ 1 + ",
                 "tenam(", Vte[1], ", ", Vte[2], ") + ",
                 paste(paste0("snam(", Vs, ")"), collapse=" + "),
                 " + deep_mod(",
                 paste(Vs, collapse=", "), ")")
  
  res <- dr(form, 
             trainX, trainY, testX, 
             oz_option = orthog_control(orthogonalize = FALSE),
             list_of_deep_models = list(
               deep_mod = deep_mod,
               snam = feature_net,
               tenam = feature_net_bi
             ),
             ...)
  
  return(res)
  
  
}

############################# SCTM #############################
sctm <- function(trainX, trainY, testX, testY,
                 ...){

  Vte <- c("latitude", "longitude")
  Vs <- setdiff(colnames(trainX), Vte)
  
  form <- paste0("y ~ 1 + ",
                 "te(", Vte[1], ", ", Vte[2], ") + ",
                 paste(paste0("s(", Vs, ")"), collapse=" + "))
  
  res <- tm(form, 
            trainX, trainY, testX, 
            oz_option = orthog_control(orthogonalize = FALSE),
            ...)
  
  return(res)
  
  
}

############################# NEAT #############################
neat <- function(trainX, trainY, testX, testY,
                 architecture,
                 ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  Vte <- c("latitude", "longitude")
  Vs <- setdiff(colnames(trainX), Vte)
  
  form <- paste0("y ~ 1 + ",
                 "tenam(", Vte[1], ", ", Vte[2], ") + ",
                 paste(paste0("snam(", Vs, ")"), collapse=" + "))
  
  res <- tm(form, 
            trainX, trainY, testX, 
            oz_option = orthog_control(orthogonalize = FALSE),
            list_of_deep_models = list(
              deep_mod = deep_mod,
              snam = feature_net,
              tenam = feature_net_bi
            ),
            ...)
  
  return(res)
  
  
}

############################# SNEAT #############################
sneat <- function(trainX, trainY, testX, testY,
                 architecture,
                 ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  Vte <- c("latitude", "longitude")
  Vs <- setdiff(colnames(trainX), Vte)
  
  form <- paste0("y ~ 1 + ",
                 "tenam(", Vte[1], ", ", Vte[2], ") + ",
                 paste(paste0("snam(", Vs, ")"), collapse=" + "),
                 " + deep_mod(",
                 paste(Vs, collapse=", "), ")")
  
  res <- tm(form, 
             trainX, trainY, testX, 
             oz_option = orthog_control(orthogonalize = FALSE),
             list_of_deep_models = list(
               deep_mod = deep_mod,
               snam = feature_net,
               tenam = feature_net_bi
             ),
             ...)
  
  return(res)
  
  
}