reticulate::use_condaenv("r-reticulate")
# contains python
# tensorflow
library(tidyverse)
library(ggplot2)
library(ggsci)
library(devtools)
library(tensorflow)
library(spatsurv)
library(survival)
library(lubridate)
library(deeptrafo)
library(deepregression)
library(mixdistreg)
library(pammtools)
library(keras)
source("helpers.R")

fstimes <- readRDS("application/Survival/data/fstimes.Rds")
fstimes$surv_times <- fstimes$S[, 1]
fstimes$status <- fstimes$S[, 2]
fstimes$status <- ifelse(fstimes$surv_times > 1000, 0, 1)
fstimes$surv_times <- ifelse(fstimes$surv_times >= 1000, 1000, fstimes$surv_times)
#admin censoring at 1000
fstimes$BoroughName <- (as.factor(fstimes$BoroughName))
fstimes$WardName <- (as.factor(fstimes$WardName))
fstimes$PropertyType <- (as.factor(fstimes$PropertyType))
fstimes$BoroughName_i <- as.integer(as.factor(fstimes$BoroughName))
fstimes$WardName_i <- as.integer(as.factor(fstimes$WardName))
fstimes$PropertyType_i <- as.integer(as.factor(fstimes$PropertyType))
fstimes_ped <- fstimes %>%
  as_ped.SpatialPointsDataFrame(Surv(surv_times, status) ~ timenumeric + 
                                  .x + .y + WardName + BoroughName +
                                  PropertyType, 
                                cut = c(1, 3, 5, 7, 10, 15, seq(20, 1000, by = 20)))

scaling <- fstimes_ped %>% group_by(id) %>% 
  summarise(timenumeric = mean(timenumeric), 
            .x = mean(.x), .y = mean(.y)) %>%
  summarise(m.timenumeric = mean(timenumeric), s.timenumeric = sd(timenumeric),
            m.x = mean(.x), s.x = sd(.x), m.y = mean(.y), s.y = sd(.y)) %>% 
  mutate(m.tend = mean(fstimes$surv_times), s.tend = sd(fstimes$surv_times))

fped <- fstimes_ped %>% 
  mutate(timenumeric = (timenumeric - scaling$m.timenumeric) / scaling$s.timenumeric,
         .x = (.x - scaling$m.x) / scaling$s.x,
         .y = (.y - scaling$m.y) / scaling$s.y,
         tend_deep = (tend - scaling$m.tend) / scaling$s.tend)

form_neat <- surv ~ 1 + snam(timenumeric) + tenam(Easting_m, Northing_m) + 
  PropertyType + BoroughName

fstimes_surv <- as.data.frame(fstimes) %>% 
  dplyr::select(surv_times, status, timenumeric, Northing_m, Easting_m, 
                WardName, BoroughName, PropertyType, WardName_i, BoroughName_i, PropertyType_i) %>%
  mutate(surv = Surv(surv_times, status)) %>%
  mutate(cens = status, status = NULL, surv_times = NULL) %>%
  mutate(timenumeric = (timenumeric - mean(timenumeric)) / sd(timenumeric),
         Easting_m = (Easting_m - mean(Easting_m)) / sd(Easting_m),
         Northing_m = (Northing_m - mean(Northing_m)) / sd(Northing_m))

feature_net <- function(x) x %>% 
  layer_dense(units = 64, activation = "relu", use_bias = TRUE) %>% 
  layer_dense(units = 12, activation = "relu", use_bias = TRUE) %>% 
  layer_dense(units = 1, use_bias = TRUE)

feature_net_bi <- function(x){ 
  
  base1 <- tf_stride_cols(x, 1) %>% 
    layer_dense(units = 32, "relu", use_bias = TRUE) %>%
    layer_dense(units = 32, "relu", use_bias = TRUE) %>%
    layer_dense(units = 16, "relu", use_bias = TRUE) %>%
    layer_dense(units = 10, activation = "relu")
  
  base2 <- tf_stride_cols(x, 2) %>% 
    layer_dense(units = 32, "relu", use_bias = TRUE) %>%
    layer_dense(units = 32, "relu", use_bias = TRUE) %>%
    layer_dense(units = 16, "relu", use_bias = TRUE) %>%
    layer_dense(units = 10, activation = "relu")
  
  tf_row_tensor(base1, base2) %>% 
    layer_dense(1, use_bias = TRUE)
  
}

deep_mod_nam <- function(x) x %>% 
  layer_dense(units = 16, activation = "relu", use_bias = TRUE) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 8, activation = "relu") %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 1)
