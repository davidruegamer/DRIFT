source("general.R")
# avoid rare categorical levels for PAM
fstimes$PropertyType[
  fstimes$PropertyType == "Houseboat (permanent dwelling) " |
    fstimes$PropertyType == "Caravan/Mobile home (permanent dwelling)"] <-
  NA
fstimes$BoroughName[fstimes$BoroughName == "City of London"] <- NA
fstimes <- fstimes[!is.na(fstimes$BoroughName) & !is.na(fstimes$PropertyType),]
fstimes$BoroughName <- (as.factor(as.character(fstimes$BoroughName)))
fstimes$PropertyType <- (as.factor(as.character(fstimes$PropertyType)))
fstimes_ped <- fstimes %>%
  as_ped.SpatialPointsDataFrame(Surv(surv_times, status) ~ timenumeric + 
                                  .x + .y + WardName +
                                  BoroughName +
                                  PropertyType, 
                                cut = c(1, 3, 5, 7, 10, 18, seq(25, 1000, by = 35)))
res <- martingale_residuals <- vector("list", 25L)
set.seed(8)
for (i in 1:25) {
  print(i)
  set.seed(i)
  train_id <- sample(1:nrow(fstimes), round(0.8 * nrow(fstimes)))
  test_id <- setdiff(1:nrow(fstimes), train_id)
  
  scaling <- fstimes_ped %>% filter(id %in% train_id) %>%
    group_by(id) %>% 
    summarise(timenumeric = mean(timenumeric), 
              .x = mean(.x), .y = mean(.y)) %>%
    summarise(m.timenumeric = mean(timenumeric), s.timenumeric = sd(timenumeric),
              m.x = mean(.x), s.x = sd(.x), m.y = mean(.y), s.y = sd(.y)) %>% 
    mutate(m.tend = mean(fstimes$surv_times), s.tend = sd(fstimes$surv_times))
  
  fped <- fstimes_ped %>% 
    mutate(timenumeric = (timenumeric - scaling$m.timenumeric) / scaling$s.timenumeric,
           .x = (.x - scaling$m.x) / scaling$s.x,
           .y = (.y - scaling$m.y) / scaling$s.y)
  
  fstimes_surv <- as.data.frame(fstimes) %>% 
    dplyr::select(surv_times, status, timenumeric, Northing_m, Easting_m, 
                  WardName, BoroughName, PropertyType) %>%
    mutate(surv = Surv(surv_times, status), BoroughName = BoroughName,
           PropertyType = PropertyType) %>%
    mutate(cens = status, status = NULL, surv_times = NULL) %>%
    mutate(timenumeric = (timenumeric - mean(timenumeric)) / sd(timenumeric),
           Easting_m = (Easting_m - mean(Easting_m)) / sd(Easting_m),
           Northing_m = (Northing_m - mean(Northing_m)) / sd(Northing_m)) 
  
  neat <- CoxphNN(form_neat, data = fstimes_surv[train_id, ],
                  optimizer = optimizer_adam(learning_rate = 0.001),
                  list_of_deep_models = list(
                    snam = feature_net,
                    tenam = feature_net_bi,
                    deep = deep_mod_nam
                  ))
  neat %>% fit(
    epochs = 250,
    verbose = FALSE,
    view_metrics = FALSE,
    batch_size = 32,
    validation_split = 0.1,
    callbacks = list(callback_reduce_lr_on_plateau(patience = 30, factor = 0.5, min_lr = 0.0001),
                     callback_early_stopping(patience = 75, restore_best_weights = T))
  )
  
  eval_ints <- get_eval_intervals(fstimes, "status", "surv_times")
  surv_probs <- matrix(0, length(test_id), length(eval_ints$eval_ints))
  data_test <- fstimes_surv[test_id, ]
  current <- data_test
  for (j in 1:ncol(surv_probs)) {
    current$surv <- matrix(eval_ints$eval_ints[j], nrow = nrow(surv_probs))
    surv_probs[, j] <- 1 - predict(neat, current, type = "cdf")
  }
  pam <- pamm(ped_status ~ s(tend) + s(timenumeric, bs = "cc") + 
                BoroughName + PropertyType +
                te(.x, .y, k = 10), data = fped %>% filter(id %in% train_id), 
              engine = "bam", method = "fREML")
  km <- pamm(ped_status ~ s(tend), data = fped %>% filter(id %in% train_id))
  fstimes_test <- fstimes[test_id,]
  fstimes_test$surv_times <- 1000L
  hazards_pam <- predict(
    pam, 
    fstimes_test %>%
      as_ped.SpatialPointsDataFrame(Surv(surv_times, status) ~ timenumeric + 
                                      .x + .y + BoroughName + PropertyType,
                                    cut = c(1, 3, 5, 7, 10, 18, seq(25, 1000, by = 35))) %>%
      mutate(timenumeric = (timenumeric - scaling$m.timenumeric) / scaling$s.timenumeric,
             .x = (.x - scaling$m.x) / scaling$s.x,
             .y = (.y - scaling$m.y) / scaling$s.y,
             offset = 0),
    type = "response")
  hazards_km <- predict(
    km, 
    fstimes_test %>%
      as_ped.SpatialPointsDataFrame(
        Surv(surv_times, status) ~ 1,
        cut = c(1, 3, 5, 7, 10, 18, seq(25, 1000, by = 35))) %>%
      mutate(offset = 0),
    type = "response")
  
  s_pam <- predictSurvProb_pamm(hazards_pam, eval_ints$eval_ints, 
                                as.data.frame(fstimes_test), pam)
  s_km <- predictSurvProb_pamm(hazards_km,  eval_ints$eval_ints, 
                               as.data.frame(fstimes_test), km)
  test_data <- data.frame(time = fstimes$surv_times[test_id], 
                          status = fstimes$status[test_id])
  ours <- rgs_score(surv_probs, as.data.frame(test_data), 
                    eval_ints$eval_ints, at = eval_ints$quartiles)
  PAM <- rgs_score(s_pam, as.data.frame(test_data), 
                   eval_ints$eval_ints, at = eval_ints$quartiles)
  KM <- rgs_score(s_km, as.data.frame(test_data), 
                  eval_ints$eval_ints, at = eval_ints$quartiles)
  res[[i]] <- list(ours = ours, PAM = PAM, KM = KM)
  
  #martingale residuals
  cumhaz <- - log(surv_probs)
  cumhaz_pam <- - log(s_pam)
  martingale <- martingale_pam <- rep(0, length(test_data$time))
  for (s in 1:length(martingale)) {
    when <- which.min((eval_ints$eval_ints - test_data$time[s])^2)
    martingale[s] <- test_data$status[s] - cumhaz[s, when]
    martingale_pam[s] <- test_data$status[s] - cumhaz_pam[s, when]
  }
  martingale_residuals[[i]] <- list(ours = martingale, pam = martingale_pam) 
}

saveRDS(res, "results/res.Rds")
saveRDS(martingale_residuals, "results/martingales.Rds")


