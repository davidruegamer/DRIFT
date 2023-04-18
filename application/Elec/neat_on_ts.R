rm(list = ls())

# start file from NEAT directory

# set path to (conda) env
conda_env <- "/Users/flipst3r/opt/anaconda3/envs/r-reticulate" 
reticulate::use_condaenv(conda_env, required = TRUE)
devtools::load_all("./neat")
#library(tensorflow)
#library(tfprobability)
#library(keras)
library(data.table)
library(caret)
library(parallel)
tf$constant(1) # check TF

## Load Data and train_mod()

ts_application <- file.path(getwd(), "application","ts_example")
data_path <- file.path(ts_application, "electricity.RDS")
source(file.path(ts_application, "utils.R"))

if (!file.exists(data_path)) {
  # prepare data
  source(file.path(ts_application, "prep_data_elec.R"))
}

electricity <- readRDS(data_path)

### Set globals and prepare for parallel across IDs

max_ep <- 1e4 # max epochs for validation
reps <- 10 # number of repeated runs for PLS std. error
cl <- makeCluster(parallel::detectCores() - 1, outfile = '')
clusterExport(cl, c("electricity", "conda_env","train_mod","max_ep","reps"))
clusterEvalQ(cl, {library(data.table)
  library(caret)})

### Executes

strt <- Sys.time()

res <- parLapply(cl, unique(electricity$d_val_tr$id), function(idd) {
  
  ## cycles through ts (one per id)
  cat("ID:", idd, "\n")
  
  reticulate::use_condaenv(conda_env, required = TRUE)
  devtools::load_all("./neat")
  
  d <- lapply(electricity, function(x) x[id == idd]) # subset
  val_tr <- d$d_val_tr
  val_tst <- d$d_val_tst
  lags <- colnames(val_tr)[grepl("^y_lag", colnames(val_tr))]
  p <- length(lags)
  
  # define model
  def_mod <- function() neat(p, type = "ls",
                             optimizer = optimizer_adam(learning_rate = 0.0001))
  m <- def_mod()
  
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
  
  # train_mod() defined in utils.R
  tst_epochs <- train_mod(m,
                          ep = max_ep, 
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
  neat_logliks <- sapply(1:reps, function(rp) {
    
    m <- def_mod() # reset model to avoid continued training
    train_mod(m, ep = tst_epochs, 
              d_tr = list(list(X_tr, y_tr), y_tr),
              d_val = list(list(X_tst, y_tst), y_tst),
              v = 0, 
              final = TRUE, s = rp)
  }, simplify = TRUE)
  
  return(neat_logliks)
  
})

stopCluster(cl)

# logLiks
res_elec <- do.call("rbind",res) # rows have IDs and cols have reps
attr(res_elec, "run_time") <- Sys.time() - strt
attr(res_elec, "session_info") <- sessionInfo()

saveRDS(res_elec, file = file.path(ts_application, "res_neat.RDS"))
