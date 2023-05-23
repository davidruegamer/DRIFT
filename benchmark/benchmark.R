############################# Loading libraries #############################
library(caret)
library(parallel)
nr_cores <- 4
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
                   testY = testY,
                   # maxEpochs = 1L,
                   architecture = NULL
                   )
      if(!is.null(ar)){
        args$architecture <- ar
      }
      
      tryNA(do.call(fun, args))
    }

    # models
    res_fold_i <- suppressMessages(suppressWarnings(
      data.frame(
        
        # deep distreg
        ddr_a = fun_w_ar(ddr, "a"),
        ddr_b = fun_w_ar(ddr, "b"),
        ddr_c = fun_w_ar(ddr, "c"),
        ddr_d = fun_w_ar(ddr, "d"),

        # deep neural trafos L-S model type        
        deepneatls_a = fun_w_ar(deepneatls, "a"),
        deepneatls_b = fun_w_ar(deepneatls, "b"),
        deepneatls_c = fun_w_ar(deepneatls, "c"),
        deepneatls_d = fun_w_ar(deepneatls, "d"),
        
        # deep neural trafos T-P model type
        # deepneattp_a = fun_w_ar(deepneattp, "a"),
        # deepneattp_b = fun_w_ar(deepneattp, "b"),
        # deepneattp_c = fun_w_ar(deepneattp, "c"),
        # deepneattp_d = fun_w_ar(deepneattp, "d"),

        # deep neural trafos interconnected model type        
        # deepneatinter_a = fun_w_ar(deepneatinter, "a"),
        # deepneatinter_b = fun_w_ar(deepneatinter, "b"),
        # deepneatinter_c = fun_w_ar(deepneatinter, "c"),
        # deepneatinter_d = fun_w_ar(deepneatinter, "d"),

        # structured additive distreg
        sadr = fun_w_ar(sadr, NULL),
        
        # semi-structured additive distreg
        ssdr_a = fun_w_ar(ssdr, "a"),
        ssdr_b = fun_w_ar(ssdr, "b"),
        ssdr_c = fun_w_ar(ssdr, "c"),
        ssdr_d = fun_w_ar(ssdr, "d"),
        
        # structured additive distreg with NAM effects
        # namdr = fun_w_ar(namdr, NULL),
        
        # semi-structured additive distreg with NAM effects
        snamdr_a = fun_w_ar(snamdr, "a"),
        snamdr_b = fun_w_ar(snamdr, "b"),
        snamdr_c = fun_w_ar(snamdr, "c"),
        snamdr_d = fun_w_ar(snamdr, "d"),
        
        # structured CTMs
        sctm = fun_w_ar(sctm, NULL),
        
        # deep CTMs
        deepctm_a = fun_w_ar(deepctm, "a"),
        deepctm_b = fun_w_ar(deepctm, "b"),
        deepctm_c = fun_w_ar(deepctm, "c"),
        deepctm_d = fun_w_ar(deepctm, "d"),
        
        # structured trafos with NAM effects
        namtm = fun_w_ar(namtm, NULL),
        # structured neural trafos L-S model type
        neatls = fun_w_ar(neatls, NULL),
        # structured neural trafos T-P model type
        # neattp = fun_w_ar(neattp, NULL),
        # structured neural trafos interconnected model type
        # neatinter = fun_w_ar(neatinter, NULL),

        # semi-structured neural trafos L-S model type
        snamtam_a = fun_w_ar(snamtam, "a"),
        snamtam_b = fun_w_ar(snamtam, "b"),
        snamtam_c = fun_w_ar(snamtam, "c"),
        snamtam_d = fun_w_ar(snamtam, "d"),
        
        # semi-structured neural trafos L-S model type
        neatls_a = fun_w_ar(neatls, "a"),
        neatls_b = fun_w_ar(neatls, "b"),
        neatls_c = fun_w_ar(neatls, "c"),
        neatls_d = fun_w_ar(neatls, "d")
        
        # semi-structured neural trafos T-P model type
        # neattp_a = fun_w_ar(neattp, "a"),
        # neattp_b = fun_w_ar(neattp, "b"),
        # neattp_c = fun_w_ar(neattp, "c"),
        # neattp_d = fun_w_ar(neattp, "d"),
        
        # semi-structured neural trafos interconnected model type
        # neatinter_a = fun_w_ar(neatinter, "a"),
        # neatinter_b = fun_w_ar(neatinter, "b"),
        # neatinter_c = fun_w_ar(neatinter, "c"),
        # neatinter_d = fun_w_ar(neatinter, "d")

        )
    ))
    
    
    
  }, mc.cores = nr_cores)
  
  return(res)
  
}

if(!dir.exists("results"))
  dir.create("results")

datas <- c("airfoil", 
           "concrete", 
           "diabetes", 
           "energy",
           "fish", 
           "forest_fire", 
           "ltfsid", 
           "naval_compressor", 
           "real", 
           "wine", 
           "yacht") 

for(nam in datas){
  
  res <- benchmark_per_dataset(nam)
  saveRDS(res, file=paste0("results/", nam, "_sadr.RDS"))
  
}

if(FALSE){
  
  # produce result table
  library(xtable)
  library(dplyr)
  rounding <- 2
  lf <- list.files("results/")
  tab_raw <- do.call("rbind", lapply(1:length(lf), function(i){
    
    table_for_data_i <- do.call("rbind", readRDS(paste0("results/", lf[i])))
    # if(lf[i] %in% c("naval_compressor.RDS",
    #                 "naval_turbine.RDS")) table_for_data_i <- table_for_data_i * 100
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
  
  colnames(tab_raw)[grepl("^neat[ls|tp|inter]", colnames(tab_raw)) & 
                      grepl("\\_[a|b|c|d]", colnames(tab_raw))] <- paste0("s", 
    colnames(tab_raw)[grepl("^neat[ls|tp|inter]", colnames(tab_raw)) & 
                        grepl("\\_[a|b|c|d]", colnames(tab_raw))] 
                      )
  
  ctr <- unique(gsub("\\_[a|b|c|d]", "", colnames(tab_raw)))
  tab_agg <- as.data.frame(
    lapply(1:length(ctr), function(i)
      apply(tab_raw[,grepl(ctr[i], colnames(tab_raw)),drop=F],1, max))
  )
  colnames(tab_agg) <- c("DDR", "SADR", "SSADR",
                         "NAMDR", "SSNAMDR", "CTM", "DCTM",
                         "NAMCTM", "SSNAMCTM", "DNEAT", "SNEAT",  
                         "SSNEAT")
  
  tab_agg <- tab_agg %>% 
    mutate_all(~ if_else(as.numeric(trimws(gsub("(.*)(\\(.*\\))", "\\1", .x))) < -100 | 
                           .x == "NA (NA)", 
                         "NC (NC)", .x))
  
  tab_agg %>% t() %>% xtable()

}