rm(list = ls())
conda_env <- ""
reticulate::use_condaenv(conda_env, required = TRUE)
devtools::load_all("./neat")
elec_RDS <- ""
electricity <- readRDS(elec_RDS)
library(tensorflow)
library(tfprobability)
library(keras)
library(data.table)
library(caret)
tf$constant(1) # check TF

train_mod <- function(mod, ep, bs = 256L, d_tr = NULL, d_val = NULL, v = 2, s = 1, final = FALSE) {
  
  ## trains a neat model and makes its final prediction on test data (if final == T)
  
  # mod: defined neat model
  # d_tr: list(list(as.matrix(X), as.matrix(y)), as.matrix(y))
  # d_val: list(list(as.matrix(X), as.matrix(y)), as.matrix(y))
  # s: seed
  # ep: iterations for model fiting
  # bs: batch size
  # v: verbose keras interface

  tensorflow::set_random_seed(s)
  
  if(is.null(d_tr)) stop("Provide training data.")
  
  if (final) {
    
    cat("Final model with ep =", ep, "\n")

    mod |> fit(x = d_tr[[1]], y = d_tr[[2]], 
               epochs = ep, 
               callbacks = list(), 
               batch_size = bs,
               view_metrics = FALSE,
               verbose = v, validation_data = NULL, validation_split = NULL)
    
    gc()
    
    # log likelihood on test
    return(- mod$evaluate(d_val[[1]], d_val[[2]])/nrow(d_val[[1]][[1]]))
    
  } else {
    
    cat("Trains model with max ep =", ep, "\n")

    hist <- mod |> fit(x = d_tr[[1]], y = d_tr[[2]],
                       epochs = ep, callbacks = list(callback_early_stopping(patience = 10, monitor = "val_logLik", restore_best_weights = T),
                                                     callback_reduce_lr_on_plateau(patience = 5, factor = 0.5, monitor = "val_logLik")),
                       batch_size = bs, 
                       view_metrics = FALSE,
                       validation_data = d_val,
                       verbose = v)
    
    gc()
    
    # optimal number of iterations on training data
    return(which.min(hist$metrics$val_logLik))
  }
}

res <- sapply(unique(electricity$d_val_tr$id)[1:5], function(idd) {
  
  ## iterate through ts (one per id)
  
  d <- lapply(electricity, function(x) x[id == idd]) # subset
  val_tr <- d$d_val_tr
  val_tst <- d$d_val_tst
  
  lags <- colnames(val_tr)[grepl("^y_lag", colnames(val_tr))]
  p <- length(lags)
  
  # define model
  m_orig <- neat(p, 
            type = "ls", 
            optimizer = optimizer_adam(learning_rate = 0.0001),
  )
  
  ## train model to determine optimal no. of epochs on test
  
  # train data (one week)
  X_tr <- as.matrix(val_tr[,..lags])
  y_tr <- matrix(val_tr$y)
  
  # validation data (one day)
  X_tst <- as.matrix(val_tst[,..lags])
  y_tst <- matrix(val_tst$y)
  
  # standardize
  preProcValues <- preProcess(X_tr, method = c("center", "scale"))
  X_tr <- predict(preProcValues, X_tr)
  X_tst <- predict(preProcValues, X_tst)
  
  tst_epochs <- train_mod(m_orig,
                          ep = 1e4, 
                          d_tr = list(list(X_tr, y_tr), y_tr),
                          d_val = list(list(X_tst, y_tst), y_tst), 
                          v = 0)
  
  ## predict on test with the previously determined no. of epochs
  
  tst_tr <- d$d_tst_tr
  tst_tst <- d$d_tst_tst
  
  # train data for final test (one week + one day)
  X_tr <- as.matrix(tst_tr[,..lags])
  y_tr <- matrix(tst_tr$y)
  
  # final test data (one day) for reporting
  X_tst <- as.matrix(tst_tst[,..lags])
  y_tst <- matrix(tst_tst$y)
  
  # standardize
  preProcValues <- preProcess(X_tr, method = c("center", "scale"))
  X_tr <- predict(preProcValues, X_tr)
  X_tst <- predict(preProcValues, X_tst)
  
  # repeated model runs to gauge stochasticity of neats
  neat_logliks <- sapply(1:10, function(rp) {
    
    m <- m_orig # reset model to avoid continued training
    train_mod(m, ep = tst_epochs, 
              d_tr = list(list(X_tr, y_tr), y_tr),
              d_val = list(list(X_tst, y_tst), y_tst),
              v = 0, 
              final = TRUE, s = rp)
  }, simplify = TRUE)
  
  return(neat_logliks)
  
})

# logLiks
colMeans(res)
apply(res, 2, sd)
