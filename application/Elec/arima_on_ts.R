rm(list = ls())

# start file from NEAT directory

pacman::p_load(furrr, scoringRules, xts, purrr, forecast, data.table)

ts_application <- file.path(getwd(), "application","Elec")
data_path <- file.path(ts_application, "electricity.RDS")

if (!file.exists(data_path)) {
  # prepare data
  source(file.path(ts_application, "prep_data_elec.R"))
}

no_cores <- parallel::detectCores() - 1
metric <- "logscore"
sub_index <- NULL # try "2"
plan(cluster, workers = no_cores)

find_arima <- function(yy, pp = 24, qq = 3, trun = NULL) {
  
  # yy: time series preferably ts() or xts()
  
  m <- try(auto.arima(y = yy,
                      stepwise = T, trace = T, max.p = 1e4, max.q = 1e4, ic = "aicc", max.P = 1e4, max.Q = 1e4,
                      approximation = T, start.p = pp, start.q = qq, allowdrift = T, allowmean = T,
                      max.order = 2e4, nmodels = 25, truncate = trun), silent = T) # ?optim()
  if (inherits(m, "try-error")){ 
    cat("Auto arima failed, but proceeds...\n")
    m <- try(Arima(yy), silent = T)
  }
  
  if (inherits(m, "try-error")){ 
    cat("Auto arima failed, but proceeds with ML...\n")
    m <- try(Arima(yy, method = "ML"), silent = T)
  }
  
  if (inherits(m, "try-error")){ 
    cat("Auto arima failed, but proceeds with CSS...\n")
    m <- try(Arima(yy, method = "CSS"), silent = T)
  }
  
  if(inherits(m, "try-error")) stop("No model converged with find_arima().")
  
  return(m)
}

make_ts <- function(yy, tt, fr = 12) {
  
  # auto.arima() reacts differently to ts() class with frequency attribute
  # than to plain numeric() in terms of seasonality detection
  # frequency 24 rather than 365*24 for hourly improves forecast
  # check https://robjhyndman.com/uwa2017/1-1-Graphics.pdf slide 19/77 for ts frequency
  
  y <- xts(yy, order.by = tt)
  attr(y, 'frequency') <- fr
  y
  #as.ts(y)
}

fit_arima <- function(mod, yy) {
  
  # takes model complexity of a arima model coming from auto.arima() and refits on new data yy 
  # mod$arma: 
  # the number of AR, MA, seasonal AR and seasonal MA coefficients, 
  # the period, the number of non-seasonal and seasonal differences.
  # maybe consider the "model" argument of Arima()
  
  m <- try(Arima(yy, order = mod$arma[c(1,6,2)], seasonal = mod$arma[c(3,7,4)],
                 include.drift = any(grepl("drift", names(coef(mod))))), silent = T)
  if (inherits(m, "try-error")){ 
    cat("Predefined arima failed, but proceeds with ML...\n")
    m <- try(Arima(yy, method = "ML", order = mod$arma[c(1,6,2)], seasonal = mod$arma[c(3,7,4)],
                   include.drift = any(grepl("drift", names(coef(mod))))), silent = T)
  }
  
  if (inherits(m, "try-error")){ 
    cat("Predefined arima failed, but proceeds with CSS...\n")
    m <- try(Arima(yy, method = "CSS", order = mod$arma[c(1,6,2)], seasonal = mod$arma[c(3,7,4)],
                   include.drift = any(grepl("drift", names(coef(mod))))), silent = T)
  }
  
  methods <- c("Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN", "Brent")
  idx <- 1
  while(inherits(m, "try-error") && (idx <= length(methods))){ 
    cat("Predefined arima failed, but proceeds with ML and method", methods[idx], "\n")
    m <- try(Arima(yy, method = "ML", optim.method = methods[idx]), silent = T)
    idx <- idx + 1
  }
  
  if(inherits(m, "try-error")) {
    message("No model converged with fit_arima().\n")
  } 
  
  m
}


forecast_arima <- function(m, d_evl, what = "logscore") {
  
  # modify forecast:::forecast.forecast_ARIMA so bootstrap draws are returned: sim = sim in line 145 and use crps_sample()
  # or return std. dev. (under gaussianity, pred$se) used to construct conf intervals and use crps_norm() instead
  
  new_func <- forecast:::forecast.forecast_ARIMA # does not find package when in package envir. so put in function envir.
  body(new_func)[[25]][[2]][[2]] <- substitute(list(method = method, model = object, 
                                                    level = level, mean = pred$pred, lower = lower, 
                                                    upper = upper, x = x, series = seriesname, 
                                                    fitted = fits, residuals = residuals.Arima(object), se = pred$se))
  
  forecast_horizon <- time(d_evl)
  for_cast <- try(new_func(m, h = length(forecast_horizon)))
  if(inherits(for_cast, "try-error")) return(xts::as.xts(rep(0, length(forecast_horizon)), order.by = forecast_horizon))
  
  if(what == "logscore"){
    logs <- sapply(seq_along(forecast_horizon), function(t_idx) {
      dnorm(x = d_evl[t_idx], mean = for_cast$mean[t_idx], sd = for_cast$se[t_idx], log=TRUE)
    })
    
    return(xts::as.xts(logs, order.by = forecast_horizon))
  }else{
    crps <- sapply(seq_along(forecast_horizon), function(t_idx) {
      crps_norm(y = d_evl[t_idx], mean = for_cast$mean[t_idx], sd = for_cast$se[t_idx])
    })
    
    return(xts::as.xts(crps, order.by = forecast_horizon))
  }
}

transform_ts <- function(d, fr2) {
  
  # from data.frame of time series to list of time series
  # time stamps might not be the same across time series so avoid tapply() and use split() in the following
  
  ts_per_id <- split(d$y, d$idfac)
  time_per_id <- split(d$t_ime, d$idfac)
  map2(ts_per_id, time_per_id, ~make_ts(.x, .y, fr = fr2))
}

run_arima <- function(data, freq = 24, metric = "crps", sub_index = NULL, trun_auto_arima = NULL) {
  
  # run auto.arima on benchmark models
  # data: list(d_val_tr = d_val_tr,
  #            d_val_tst = d_val_tst,
  #            d_tst_tr = d_tst_tr,
  #            d_tst_tst = d_tst_tst)
  # freq (int): frequency of the ts in data which is automatically recognized by auto.arima() for seasonality detection
  # metric: either "crps" or "logscore"
  # sub_index: performs subsetting of the time series for quick mock runs
  
  # validation train
  val_tr <- transform_ts(data$d_val_tr, freq)
  
  # validation test
  val_tst <- transform_ts(data$d_val_tst, freq)
  
  # test train
  tst_tr <- transform_ts(data$d_tst_tr, freq)
  
  # test test
  tst_tst <- transform_ts(data$d_tst_tst, freq)
  
  # for quick mock runs
  if (!is.null(sub_index)) {
    val_tr <- val_tr[1:sub_index]
    val_tst <- val_tst[1:sub_index]
    tst_tr <- tst_tr[1:sub_index]
    tst_tst <- tst_tst[1:sub_index]
  }
  #browser()
  
  # Try different starting values for stepwise search of auto.arima(p,q) depending on frequency
  max_hor <- max(c(sapply(tst_tst, length), sapply(val_tst, length))) # try order of lags depending on forecast horizon
  param_grid <- expand.grid(p_lag = c(floor(max_hor/2), max_hor), q_lag = c(0,3))
  param_grid_list <- split(param_grid, seq_len(nrow(param_grid)))
  
  # try each parameter constellation for each ts in the data
  # replace map() by future_map() for a parallel version
  arima_models <- lapply(param_grid_list, function(param_set) {
    future_map(val_tr, ~find_arima(.x, pp = param_set$p_lag, qq = param_set$q_lag, trun = trun_auto_arima), .progress = T)
  })
  
  # see how each model performs on the validation test data
  scores_on_val_test <- lapply(arima_models, function(arima_models_param_set) {
    map2(arima_models_param_set, val_tst, ~forecast_arima(.x, .y, what = metric))
  })
  
  # average across the forecasting horizon
  scores_per_param_set <- sapply(scores_on_val_test, function(scores_on_test_param_set){
    mean(do.call("cbind", scores_on_test_param_set), na.rm = T)
  })
  
  # find best arimas on validation and refit them on the final training set
  if (metric == "logscore") {
    best_arimas <- arima_models[[which.max(scores_per_param_set)]]
  } else { # crps should be minimized
    best_arimas <- arima_models[[which.min(scores_per_param_set)]]
  }
  
  fitted_best_arimas <- map2(best_arimas, tst_tr, ~fit_arima(.x, .y))
  scores_on_test <- map2(fitted_best_arimas, tst_tst, ~forecast_arima(.x, .y, what = metric))
  
  # score on test
  s <- mean(do.call("cbind", scores_on_test), na.rm = T)
  cat("Score on test:", s, "\n")
  s
}

strt <- Sys.time()
data <- readRDS(data_path)
args <- list(data = data, freq = 24, metric = metric, sub_index = sub_index)
res_elec <- do.call("run_arima", args)

attr(res_elec, "metric") <- metric
attr(res_elec, "run_time") <- Sys.time() - strt
attr(res_elec, "session_info") <- sessionInfo()

saveRDS(res_elec, file = file.path(ts_application, "res_arima.RDS"))
