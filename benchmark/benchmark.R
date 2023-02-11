############################# Loading libraries #############################
library(caret)
library(parallel)
nr_cores <- 5
tryNA <- function(expr) tryCatch(expr, error = function(e) NA)
############################# Data loader #################################
data_reader <- function(
  name = c("airfoil", "concrete", "diabetes", "energy",
           "fish", "forest_fire", "ltfsid", 
           "naval_compressor", "naval_turbine", 
           "real", "wine", "yacht") 
    ){

    name <- match.arg(name)
    data <- read.table(paste0("benchmark_data/", name, ".data"))
    return(data)
  
}
########################### Benchmark function #############################
benchmark_per_dataset <- function(name, folds = 10){
  
  data <- data_reader(name)
  
  X <- data[,1:(ncol(data)-1)]
  # exclude columns with not enough unique values
  X <- X[,which(apply(X, 2, function(x) length(unique(x)))>10)]
  X <- as.data.frame(scale(X))
  y <- data[,ncol(data)]
  
  set.seed(1)
  folds <- createFolds(y, k = folds)
  
  res <- mclapply(folds, function(testind){
    
    # source models within apply to allow for parallelization
    source("../common/models.R")
    
    # data
    trainind <- setdiff(1:nrow(X), testind)
    trainX <- X[trainind,]
    trainY <- as.numeric(y[trainind])
    testX <- X[testind,]
    testY <- as.numeric(y[testind])
    
    fun_w_ar <- function(fun, ar=NULL)
    {
      
      args <- list(trainX = trainX,
                   trainY = trainY,
                   testX = testX,
                   testY = testY)
      if(!is.null(ar)) args$architecture <- ar
      
      tryNA(do.call(fun, args))
    }

    # models
    res_fold_i <- suppressMessages(suppressWarnings(
      data.frame(
        
        ddr_a = fun_w_ar(ddr, "a"),
        ddr_b = fun_w_ar(ddr, "b"),
        ddr_c = fun_w_ar(ddr, "c"),
        ddr_d = fun_w_ar(ddr, "d"),
        
        sadr = fun_w_ar(sadr, NULL),
        
        ssdr_a = fun_w_ar(ssdr, "a"),
        ssdr_b = fun_w_ar(ssdr, "b"),
        ssdr_c = fun_w_ar(ssdr, "c"),
        ssdr_d = fun_w_ar(ssdr, "d"),
        
        namdr = fun_w_ar(namdr, NULL),
        
        snamdr_a = fun_w_ar(snamdr, "a"),
        snamdr_b = fun_w_ar(snamdr, "b"),
        snamdr_c = fun_w_ar(snamdr, "c"),
        snamdr_d = fun_w_ar(snamdr, "d"),
        
        sctm = fun_w_ar(sctm, NULL),
        
        neat = fun_w_ar(neat, NULL),
        neat2 = fun_w_ar(neat2, NULL),
        neat3 = fun_w_ar(neat3, NULL),

        sneat_a = fun_w_ar(sneat, "a"),
        sneat_b = fun_w_ar(sneat, "b"),
        sneat_c = fun_w_ar(sneat, "c"),
        sneat_d = fun_w_ar(sneat, "d"),
        
        sneat2_a = fun_w_ar(sneat2, "a"),
        sneat2_b = fun_w_ar(sneat2, "b"),
        sneat2_c = fun_w_ar(sneat2, "c"),
        sneat2_d = fun_w_ar(sneat2, "d"),
        
        sneat3_a = fun_w_ar(sneat3, "a"),
        sneat3_b = fun_w_ar(sneat3, "b"),
        sneat3_c = fun_w_ar(sneat3, "c"),
        sneat3_d = fun_w_ar(sneat3, "d")

        )
    ))
    
    
    
  }, mc.cores = nr_cores)
  
  return(res)
  
}

if(!dir.exists("results"))
  dir.create("results")

datas <- c("airfoil", "concrete", "diabetes", "energy",
           "fish", "forest_fire", "ltfsid", 
           "naval_compressor", "naval_turbine", 
           "real", "wine", "yacht") 

for(nam in datas){
  
  res <- benchmark_per_dataset(nam)
  saveRDS(res, file=paste0("results/", nam, ".RDS"))
  
}

# produce result table
library(xtable)
library(dplyr)
rounding <- 2
lf <- list.files("results/")
tab_raw <- do.call("rbind", lapply(1:length(lf), function(i){
  
  table_for_data_i <- do.call("rbind", readRDS(paste0("results/", lf[i])))
  if(lf[i] %in% c("naval_compressor.RDS",
                  "naval_turbine.RDS")) table_for_data_i <- table_for_data_i * 100
  means <- apply(table_for_data_i, 2, mean)
  sds <- apply(table_for_data_i, 2, sd)
  df <- as.data.frame(t(paste0(signif(means, rounding), " (", signif(sds, rounding), ")")))
  rownames(df) <- gsub("(.*)\\.RDS", "\\1", lf[i])
  colnames(df) <- names(means)
  return(df)
  
})) 

dsn <- tools::toTitleCase(rownames(tab_raw))
dsn[dsn=="Forest_fire"] <- "ForestF"
dsn[dsn=="Naval_compressor"] <- "Naval"

rownames(tab_raw) <- dsn

tab_raw %>% xtable()
