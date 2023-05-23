rgs_score <- function(surv_probs, data_test, eval_times, status = "status", 
                      time = "time", formula = Surv(time, status) ~ 1, 
                      at = eval_times, integrated = TRUE, adjust_intlen = TRUE) {
  survprobs <- surv_probs
  delta <- data_test[["status"]] == 1
  eval_matrix <- matrix(eval_times, nrow = nrow(survprobs), ncol = ncol(survprobs), byrow = T)
  #survprobs <- surv_probs[delta, ]
  WKM = pec::ipcw(Surv(time, status)~.,
                  data = data_test, method = "marginal", 
                  times=eval_times,
                  subjectTimes = data_test[[time]])
  follow_up <- as_ped(data_test, formula, cut = eval_times)
  possible_wgs <-  WKM$IPCW.times
  names(possible_wgs) <- unique(follow_up$interval)
  weight <- follow_up %>% group_by(id) %>% summarize(z = last(interval)) %>%
    mutate(zz = possible_wgs[z]) %>% pull(zz) %>% unname()
  wgs <- matrix(weight, nrow = length(weight), ncol = ncol(surv_probs))
  data_test <- data_test %>% mutate(id = 1:nrow(data_test))
  follow_up <- as_ped(data_test %>% mutate(time = max(eval_times)), 
                      formula, cut = eval_times)
  follow_up <- follow_up %>%
    inner_join(data_test, by = "id") %>% 
    mutate(ped_status = ifelse(.data[[time]] > tend, 1, 0))
  actual <- follow_up %>% pull(ped_status) %>% 
    matrix(nrow = nrow(survprobs), ncol = ncol(survprobs), byrow = TRUE)
  m_delta <- matrix(delta, nrow(survprobs), ncol(survprobs))
  I1 <- apply(eval_matrix, 2, ">=", data_test$time) * 1
  I1 <- I1 * m_delta
  I2 <- apply(eval_matrix, 2, "<", data_test$time) * 1
  bs <- (surv_probs^2 * I1) / wgs + ((1 - surv_probs)^2 * I2) / wgs
  if (integrated) {
    intlen <- (c(0, eval_times) - lag(c(0, eval_times)))[-1] 
    bs_ <- bs #* matrix(intlen, nrow = nrow(bs), ncol = ncol(bs), byrow = TRUE)
    ibs <- bs_
    if (adjust_intlen) {
      for (s in 1:nrow(bs_)) {
        ibs[s,] <- cumsum(bs_[s, ] * (intlen / mean(intlen)))
      } 
    } else {
      for (s in 1:nrow(bs_)) {
        ibs[s,] <- cumsum(bs_[s, ])
      } 
    }
    ibs <- ibs / (matrix(1:length(eval_times), nrow = nrow(ibs), 
                         ncol = ncol(ibs), byrow = TRUE))
    if (length(at) > 1) {
      apply(ibs[, eval_times %in% at], 2, mean)
    } else {
      mean(ibs[, eval_times %in% at])
    }
  } else {
    if (length(at) > 1) {
      apply(bs[, eval_times %in% at], 2, mean)
    } else {
      mean(bs[, eval_times %in% at])
    }
  }
}

get_eval_intervals = function(data, status = "status", time = "time") {
  e_time <- quantile(data[[time]][data[[status]] == 1], c(.25, .5, .75))
  max_time <- quantile(data[[time]][data[[status]] == 1], c(.8))
  eval_ints <- sort(c(e_time, seq(0.01, max_time, length.out = 250L)))
  list(eval_ints = eval_ints, quartiles = e_time)
}

predictSurvProb_pamm <- function(hazards, times, newdata, pam) {
  trafo_args <- pam[["trafo_args"]]
  id_var <- trafo_args[["id"]]
  brks       <- trafo_args[["cut"]]
  if ( max(times) > max(brks) ) {
    stop("Cannot predict beyond the last time point used during model estimation.
        Check the 'times' argument.")
  }
  ni <- nrow(newdata)
  tend <- as.numeric(unique(pam$model$tend))
  if (is.null(newdata[[id_var]])) newdata[[id_var]] <- 1:nrow(newdata)
  pred_frame <- data.frame(
    id = rep(newdata[[id_var]], each = length(tend)), 
    tend = rep(tend, ni), 
    pred = hazards)
  ped_times <- sort(unique(union(c(0, brks), times)))
  # extract relevant intervals only, keeps data small
  ped_times <- ped_times[ped_times <= max(times)]
  # obtain interval information
  ped_info <- get_intervals(brks, ped_times[-1])
  # add adjusted offset such that cumulative hazard and survival probability
  # can be calculated correctly
  ped_info[["intlen"]] <- c(ped_info[["times"]][1], diff(ped_info[["times"]]))
  newdata <- combine_df(ped_info, newdata)
  env_times <- times
  newdata <- inner_join(newdata, pred_frame, by = c("id" = "id", "tend" = "tend"))
  newdata[["intlen"]] <- recompute_intlen(newdata)
  newdata <- newdata %>%
    arrange(.data$id, .data$times) %>%
    group_by(.data$id) %>%
    mutate(surv = exp(-cumsum((.data$pred * .data$intlen)))) %>%
    ungroup() %>%
    filter(.data[["times"]] %in% env_times)
  surv <- matrix(newdata$surv, nrow = ni, 
                 ncol = length(times), byrow = TRUE)
  surv
}

recompute_intlen <- function(data) {
  lagged_times <- as.numeric(dplyr::lag(data$times))
  lagged_times[1] <- 0
  intlen <- data$times - lagged_times
  intlen[intlen < 0] <- intlen[1]
  as.numeric(intlen)
}

