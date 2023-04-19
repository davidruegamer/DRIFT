library(argparse)
library(data.table)
library(dplyr)

train_mod <- function(mod, ep, bs = 256L, d_tr = NULL, d_val = NULL, v = 0, s = 1, final = FALSE) {
  
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
    
    # model needs to be untrained
    stopifnot(any(sapply(get_weights(mod), sum) == 0))
    
    mod |> fit(x = d_tr[[1]], y = d_tr[[2]], 
               epochs = ep, 
               callbacks = list(), 
               batch_size = bs,
               view_metrics = FALSE,
               verbose = v, validation_data = NULL, validation_split = NULL)
    gc()
    
    # log likelihood on test; d_val[[2]] is ignored, takes d_val[[1]][[2]] as y
    log_score <- - mod$evaluate(d_val[[1]], d_val[[2]])/nrow(d_val[[1]][[1]])
    
    return(log_score)
    
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


create_lags <- function(dds, check_obs = TRUE, na_omit = TRUE, lags = NULL) {
  
  # take data.table in wide, format it to long and append lags
  # long format is prefered owing to mgcv formula interface
  
  if (is.null(lags)) stop("Need predefined number of max. lags.")
  
  dds <- data.table::melt(dds, id.vars = "ds")
  setnames(dds, old = c("variable","ds","value"), new = c("id","t_ime","y"))
  d_lagged <- dds[,  shift(.SD, lags, NA, "lag", TRUE), .SDcols = "y", by = id]
  lag_idx <- grep("_lag_",names(d_lagged))
  lag_s <- names(d_lagged)[lag_idx]
  
  ## TOO MEMORY HEAVY
  #dds <- cbind(dds, d_lagged[,lag_s, with = FALSE]); rm(d_lagged) # merge() does not work owing to multiple lines with same id but no time stamp
  dds <- setDT(c(as.list(dds),as.list(d_lagged))); rm(d_lagged) # memory friendly 
  if (sum(colnames(dds) == "id") > 1) dds$id <- NULL # cbind removes duplicate
  
  if (na_omit) dds <- na.omit(dds) # lags give NAs, watch out with holidays
  nlid <- nlevels(dds$id)
  if (check_obs) stopifnot(length(unique(table(dds$t_ime))) == 1) # analyses needs same no. of obs per unit and time point
  no_hours <- length(dds$t_ime)/nlid
  dds[, time_idx := 0:(no_hours - 1), by = id] # for smooth terms, watch out with holidays
  #attr(dds, "lags") <- lag_s
  
  dds
}

equip_d <- function(dd) {
  
  # equips data set with time features, lags and everything else needed for training and testing
  
  # dd: data.table with "ds" (POSIXct) variable and every other column is asssumed 
  #     to be an individual ts
  
  dd[, da_y := factor(weekdays(t_ime, abbreviate = T))]
  dd[, hou_r := factor(hour(t_ime))]
  dd[, mont_h := factor(month(t_ime))]
  dd[, id_hour := interaction(id, hou_r)]
  dd[, id_day := interaction(id, da_y)]
  dd[, id_month := interaction(id, mont_h)]
  
  dd[, da_y := as.integer(da_y) - 1]
  dd[, hou_r := as.integer(hou_r) - 1]
  dd[, mont_h := as.integer(mont_h) - 1]
  dd[, id_hour := as.integer(id_hour) - 1]
  dd[, id_day := as.integer(id_day) - 1]
  dd[, id_month := as.integer(id_month) - 1]
  dd[, idfac := id]
  dd[, id := as.integer(id) - 1]
  
  dd
}