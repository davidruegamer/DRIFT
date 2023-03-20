############################# Loading libraries #############################
library(deeptrafo)
library(Metrics)
devtools::load_all("../neat")

############################# Architectures #############################
get_deep_mod <- function(architecture = c("a", "b", "c", "d"),
                         with_head = TRUE){
  
  if(is.null(architecture)) return(NULL)
  
  architecture <- match.arg(architecture)
  if(with_head)
  {
    deep_mod <- switch(architecture,
                       a = function(x) x %>% 
                         layer_dense(units = 100, activation = "relu", use_bias = FALSE) %>%
                         layer_dropout(rate = 0.1) %>% 
                         layer_dense(units = 1, activation = "linear"),
                       b = function(x) x %>% 
                         layer_dense(units = 100, activation = "relu", use_bias = FALSE) %>%
                         layer_dropout(rate = 0.1) %>% 
                         layer_dense(units = 100, activation = "relu") %>%
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
  }else{
    deep_mod <- switch(architecture,
                       a = function(x) x %>% 
                         layer_dense(units = 100, activation = "relu", use_bias = FALSE) %>%
                         layer_dropout(rate = 0.1),
                       b = function(x) x %>% 
                         layer_dense(units = 100, activation = "relu", use_bias = FALSE) %>%
                         layer_dropout(rate = 0.1) %>% 
                         layer_dense(units = 100, activation = "relu") %>%
                         layer_dropout(rate = 0.1),
                       c = function(x) x %>% 
                         layer_dense(units = 20, activation = "relu", use_bias = FALSE) %>%
                         layer_dropout(rate = 0.1),
                       d = function(x) x %>% 
                         layer_dense(units = 20, activation = "relu", use_bias = FALSE) %>%
                         layer_dropout(rate = 0.1) %>% 
                         layer_dense(units = 20, activation = "relu") %>%
                         layer_dropout(rate = 0.1)
    )
  }
  
  return(deep_mod)
}

form_generator <- function(colnms, 
                           teterms = c("latitude", "longitude"), 
                           nam=FALSE,
                           add_deep=FALSE,
                           deep_only=FALSE)
{
  
  form <- "~ 1"
  
  if(!deep_only){
    
    if(teterms[1]%in%colnms & teterms[2]%in%colnms){
      Vs <- setdiff(colnms, teterms)
      if(nam){
        spatial <- paste0("tenam(", teterms[1], ", ", teterms[2], ") + ")
      }else{
        spatial <- paste0("te(", teterms[1], ", ", teterms[2], ") + ") 
      }
    }else{
      Vs <- colnms
      spatial <- ""
    }
    
    if(!nam){
      form <- paste0(form,
                     " + ",
                     spatial,
                     paste(paste0("s(", Vs, ")"), collapse=" + "))
    }else{
      form <- paste0(form,
                     " + ",
                     spatial,
                     paste(paste0("snam(", Vs, ")"), collapse=" + "))
    }

  }else{
    add_deep = TRUE
  }
    
  if(add_deep)
    form <- paste0(form, " + deep_mod(",
                   paste(colnms, collapse=", "), ")")
  
  
  return(form)
  
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
  ll <- mean(log_score(mod, data = testX, this_y = matrix(testY)))
  
  rm(mod); gc()
  
  return(ll)
  
}

#################### Generic (Deep) Transformation Model ######################

tm <- function(formula, trainX, trainY, testX, testY,
               deep_mod_list = NULL,
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
  
  args <- list(formula = as.formula(formula),
               data = data,
               latent_distr = family,
               list_of_deep_models = deep_mod_list,
               optimizer = optimizer,
               orthog_options = oz_option)
  
  ellips <- list(...)
  args <- c(args, ellips)
  
  mod <- do.call("deeptrafo", args)
  
  mod %>% fit(epochs = maxEpochs, early_stopping = TRUE, 
              patience = patience, verbose = verbose)
  
  # res <- mod %>% predict(testX)
  ll <- logLik(mod, newdata = cbind(testX, y = testY))/nrow(testX)
  
  rm(mod); gc()
  
  return(ll)
  
}

neat_generic <- function(trainX, trainY, testX, testY,
                         architecture, type,
                         addnam = TRUE,
                         optimizer = optimizer_adam(),
                         maxEpochs = 10000,
                         patience = 250, 
                         verbose = FALSE,
                         ...)
{
  
  dim_features <- ncol(trainX)

  if(!is.null(architecture)){
    
    deep_mod_ <- get_deep_mod(architecture, with_head = FALSE)
  
    if(addnam)
      deep_mod <- function(x) 
        tf$concat(list(deep_mod_(x),
                       feature_specific_network()(x)),
                  axis=1L
      ) else deep_mod <- deep_mod_
        
    mod <- neat(
      dim_features,
      net_x_arch_trunk = deep_mod,
      type = type,
      optimizer = optimizer
    )
    
  }else{
   
    mod <- sneat(
      dim_features,
      type = type,
      optimizer = optimizer
    ) 
    
  }

  mod %>% fit(x = list(as.matrix(trainX), matrix(trainY)),
              y = matrix(trainY),
              epochs = maxEpochs, 
              callbacks = callback_early_stopping(
                patience = patience, restore_best_weights = TRUE, 
                monitor = "val_logLik"
              ),
              view_metrics = FALSE,
              verbose = verbose,
              validation_split = 0.1
              )
  
  # res <- mod %>% predict(testX)
  # ll <- mod$test_step(list(list(tf$constant(as.matrix(testX), dtype = "float32"),
  #                               tf$constant(matrix(testY), dtype = "float32")), 
  #                          matrix(testY))
  #                     )$logLik$numpy()
  ll <- - mod$evaluate(list(as.matrix(testX),
                            matrix(testY)),
                       matrix(testY))/nrow(testX)
  
  rm(mod); gc()
  
  return(ll)
  
}

neatls <- function(...) neat_generic(type = "ls", ...) 
neattp <- function(...) neat_generic(type = "tp", ...) 
neatinter <- function(...) neat_generic(type = "inter", ...) 

deepneatls <- function(...) neat_generic(type = "ls", addnam = FALSE, ...) 
deepneattp <- function(...) neat_generic(type = "tp", addnam = FALSE, ...) 
deepneatinter <- function(...) neat_generic(type = "inter", addnam = FALSE, ...) 

############################# DDR #############################
ddr <- function(trainX, trainY, testX, testY,
                architecture,
                ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  form <- form_generator(colnames(trainX), deep_only = TRUE)
  
  res <- dr(form, deep_mod_list = list(deep_mod = deep_mod),
            trainX = trainX, trainY = trainY, 
            testX = testX, testY = testY,
            ...)
  
  return(res)
  
  
}
############################# SADR #############################
sadr <- function(trainX, trainY, testX, testY,
                 ...){
  
  form <- form_generator(colnames(trainX), nam = FALSE, add_deep = FALSE)
  
  res <- dr(form, trainX, trainY, testX, testY, ...)
  
  return(res)
  
  
}

############################# SSDR #############################
ssdr <- function(trainX, trainY, testX, testY,
                 architecture,
                 ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  form <- form_generator(colnames(trainX), nam = FALSE, add_deep = TRUE)
  
  res <- dr(form, deep_mod_list = list(deep_mod = deep_mod),
             trainX, trainY, testX, testY, 
             oz_option = orthog_control(orthogonalize = FALSE),
             ...)
  
  return(res)
  
  
}

############################# NAMDR #############################
namdr <- function(trainX, trainY, testX, testY,
                  architecture,
                  ...){
  
  form <- form_generator(colnames(trainX), nam = TRUE, add_deep = FALSE)

  res <- dr(form, 
            trainX, trainY, testX, testY,
            oz_option = orthog_control(orthogonalize = FALSE),
            deep_mod_list = list(
              # deep_mod = deep_mod,
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
  
  form <- form_generator(colnames(trainX), nam = TRUE, add_deep = TRUE)
  
  res <- dr(form, 
            trainX, trainY, testX, testY,
            oz_option = orthog_control(orthogonalize = FALSE),
            deep_mod_list = list(
              deep_mod = deep_mod,
              snam = feature_net,
              tenam = feature_net_bi
            ),
            ...)
  
  return(res)
  
  
}

############################# SCTM #############################
deepctm <- function(trainX, trainY, testX, testY,
                    architecture,
                    ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  form <- paste0("y ", 
                 form_generator(colnames(trainX), deep_only = TRUE))
  
  res <- tm(form, 
            trainX, trainY, testX, testY,
            oz_option = orthog_control(orthogonalize = FALSE),
            deep_mod_list = list(
              deep_mod = deep_mod),
            ...)
  
  return(res)
  
  
}
############################# SCTM #############################
sctm <- function(trainX, trainY, testX, testY,
                 ...){

  form <- paste0("y ", 
                 form_generator(colnames(trainX), nam = FALSE, add_deep = FALSE))
  
  res <- tm(form, 
            trainX, trainY, testX, testY,
            oz_option = orthog_control(orthogonalize = FALSE),
            ...)
  
  return(res)
  
  
}

############################# NAM+TM #############################
namtm <- function(trainX, trainY, testX, testY,
                  architecture,
                  ...){
  
  # deep_mod <- get_deep_mod(architecture)
  
  form <- paste0("y ", 
                 form_generator(colnames(trainX), nam = TRUE, add_deep = FALSE))
  
  res <- tm(form, 
            trainX, trainY, testX, testY, 
            oz_option = orthog_control(orthogonalize = FALSE),
            deep_mod_list = list(
              # deep_mod = deep_mod,
              snam = feature_net,
              tenam = feature_net_bi
            ),
            ...)
  
  return(res)
  
  
}

############################# SSNAM+TM #############################
snamtam <- function(trainX, trainY, testX, testY,
                    architecture,
                    ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  form <- paste0("y ", 
                 form_generator(colnames(trainX), nam = FALSE, add_deep = TRUE))
  
  res <- tm(form, 
            trainX, trainY, testX, testY,
            oz_option = orthog_control(orthogonalize = FALSE),
            deep_mod_list = list(
              deep_mod = deep_mod,
              snam = feature_net,
              tenam = feature_net_bi
            ),
            ...)
  
  return(res)
  
  
}
