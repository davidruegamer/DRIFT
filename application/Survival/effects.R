source("general.R")
neat <- CoxphNN(form_neat, data = fstimes_surv,
                  optimizer = optimizer_adam(learning_rate = 0.001),
                  list_of_deep_models = list(
                    snam = feature_net,
                    tenam = feature_net_bi,
                    deep = deep_mod_nam
                  ))

neat %>% fit(
  epochs = 250,
  verbose = TRUE,
  view_metrics = FALSE,
  batch_size = 32,
  val_split = 0.1#,
  #callbacks = list(callback_reduce_lr_on_plateau(patience = 15),
  #                 callback_early_stopping(patience = 33, restore_best_weights = T))
)

test_grid <- fstimes_surv[1:1000, ]
test_grid <- test_grid %>% 
  mutate(timenumeric = seq(min(fstimes_surv$timenumeric), 
                           max(fstimes_surv$timenumeric), length.out = 1000L), 
         Northing_m = 0, Easting_m = 0) %>%
  mutate(BoroughName_i = 0, WardName_i = 0, PropertyType_i = 0) %>%
  mutate(BoroughName = fstimes_surv$BoroughName[1], WardName = fstimes_surv$WardName[1], PropertyType = fstimes_surv$PropertyType[1])
test_grid$pred <- predict(neat, test_grid, type = "shift")
test_grid$timenumeric <- seq(min(fstimes$timenumeric), 
                             max(fstimes$timenumeric), 
                             length.out = 1000L)

test_grid_xy <- expand.grid(
  x = seq(min(fstimes_surv$Easting_m), max(fstimes_surv$Easting_m), length.out = 75L),
  y = seq(min(fstimes_surv$Northing_m), max(fstimes_surv$Northing_m), length.out = 75L))
test_grid_coords <- fstimes_surv[sample(1:nrow(fstimes), 75 * 75, replace = TRUE), ]
test_grid_coords <- test_grid_coords %>% 
  mutate(timenumeric = 0, Easting_m = test_grid_xy$x, Northing_m = test_grid_xy$y) %>%
  mutate(BoroughName_i = 0, WardName_i = 0, PropertyType_i = 0) %>%
  mutate(BoroughName = fstimes_surv$BoroughName[1], WardName = fstimes_surv$WardName[1], PropertyType = fstimes_surv$PropertyType[1])
test_grid_coords$pred <- predict(neat, test_grid_coords, type = "shift")

mask <- matrix(1, 75 * 75, 2)
for (ii in 1:nrow(mask)) {
  test_grid_coords$Easting_m[ii]
  
  if (test_grid_coords$Easting_m[[ii]] < 0) {
    y <- fstimes_surv$Northing_m[fstimes_surv$Easting_m < (test_grid_coords$Easting_m[ii] + 0.5)] 
  } else {
    y <- fstimes_surv$Northing_m[fstimes_surv$Easting_m > (test_grid_coords$Easting_m[ii] - 0.5)] 
  }
  if (test_grid_coords$Northing_m[[ii]] < 0) {
    x <- fstimes_surv$Easting_m[fstimes_surv$Northing_m < (test_grid_coords$Northing_m[ii] + 0.5)] 
  } else {
    x <- fstimes_surv$Easting_m[fstimes_surv$Northing_m > (test_grid_coords$Northing_m[ii] - 0.5)] 
  }
  y_range <- c(min(y), max(y))
  x_range <- c(min(x), max(x))
  mask[ii, 1] <- (test_grid_coords$Easting_m[[ii]] > x_range[1]) & (test_grid_coords$Easting_m[[ii]] < x_range[2])
  mask[ii, 2] <- (test_grid_coords$Northing_m[[ii]] > y_range[1]) & (test_grid_coords$Northing_m[[ii]] < y_range[2])
}

ma <- mask
test_grid_coords <- test_grid_coords %>% 
  mutate(mask = ma[, 1] * ma[, 2]) %>%
  mutate(pred = ifelse(mask, pred, NA))

pam <- pamm(ped_status ~ s(tend) + s(timenumeric, bs = "cc") + 
              BoroughName + PropertyType +
              te(.x, .y, k = 10), data = fped, engine = "bam", 
            method = "fREML")
test_grid_pam <- fped[1:1000, ] %>% 
  mutate(timenumeric = seq(min(fstimes_surv$timenumeric), 
                           max(fstimes_surv$timenumeric), 
                           length.out = 1000L),
         .x = 0, .y = 0, 
         tend = 0,
         offset = 0,
         BoroughName = fped$BoroughName[1], 
         PropertyType = fped$PropertyType[1])
test_grid_pam$pred <- predict(pam, test_grid_pam)

test_grid_coords_pam <- fped[1:nrow(test_grid_coords), ] %>%
  mutate(tend = 0, 
         timenumeric = 0,
         .x = test_grid_coords$Easting_m, 
         .y = test_grid_coords$Northing_m,
         BoroughName = fped$BoroughName[1],
         PropertyType = fped$PropertyType[1])
test_grid_coords_pam$pred <- 
  predict(pam, test_grid_coords_pam)
test_grid_coords_pam <- test_grid_coords_pam %>%
  mutate(pred = ifelse(is.na(test_grid_coords$pred), NA, pred))

grid_time <- rbind(
  data.frame(daytime = test_grid$timenumeric, method = "DRIFT", 
             Estimate = test_grid$pred) %>%
    mutate(Estimate = Estimate - min(Estimate)),
  data.frame(daytime = test_grid$timenumeric, method = "PAM",
             Estimate = test_grid_pam$pred) %>%
    mutate(Estimate = Estimate - min(Estimate))
)

test_grid_coords <- test_grid_coords %>%
  mutate(pred = pred - min(pred, na.rm = T))
test_grid_coords_pam <- test_grid_coords_pam %>%
  mutate(pred = pred - min(pred, na.rm = T))

p2 <- ggplot(grid_time, aes(daytime, Estimate, col = method)) +
  geom_path() + theme_bw() + xlab("Daytime (hours)") +
  ylab("Partial Effect") + 
  guides(col = guide_legend("Method")) +
  scale_x_continuous(breaks = seq(0, 24, by = 4))


p3 <- ggplot(test_grid_coords, aes(Easting_m, Northing_m, z = pred)) +
  geom_raster(aes(fill = pred)) + 
  #geom_contour(color = "white", bins = 8, alpha = 0.5) +
  scale_fill_viridis_c() +
  coord_cartesian(expand = FALSE) +
  theme_bw() + 
  ggtitle("DRIFT") +
  xlab("Easting") +
  ylab("Northing") +
  theme(panel.grid = element_blank(), 
        panel.border = element_blank(),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        legend.position = "none") 

p4 <- ggplot(test_grid_coords_pam, aes(.x, .y, z = pred)) +
  geom_raster(aes(fill = pred)) + 
  #geom_contour(color = "white", bins = 8, alpha = 0.5) +
  scale_fill_viridis_c() +
  coord_cartesian(expand = FALSE) +
  theme_bw() + 
  ggtitle("PAM") +
  xlab("Easting") +
  ylab("Northing") +
  theme(panel.grid = element_blank(), 
        panel.border = element_blank(),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        legend.position = "bottom") +
  guides(fill = guide_legend("Partial Effect"))
  
test_grid_surv <- fstimes_surv[c(1:4000, 1:5000), ] %>%
  mutate(timenumeric = rep(c(4, 6, 8, 11, 13, 15, 18, 20, 23), each = 1000L), 
         Northing_m = 0, Easting_m = 0) %>%
  mutate(BoroughName_i = 0, WardName_i = 0, PropertyType_i = 0) %>%
  mutate(BoroughName = fstimes_surv$BoroughName[1], 
         WardName = fstimes_surv$WardName[1], 
         PropertyType = fstimes_surv$PropertyType[1],
         method = "DRIFT",
         obs = rep(1:9, each = 1000L)) %>%
  mutate(timenumeric = (timenumeric - 
                          mean(fstimes$timenumeric)) / sd(fstimes$timenumeric))
test_grid_surv$surv[, 1] <- rep(seq(min(fstimes_surv$surv[, 1]), 
                                    max(fstimes_surv$surv[, 1]),
                                    length.out = 1000L), 9)
test_grid_surv$surv[, 2] <- 1
test_grid_surv$sp <- 1 - predict(neat, test_grid_surv, type = "cdf")
test_grid_surv <- test_grid_surv %>%
  select(method, timenumeric, sp, obs)

test_grid_surv_pam <- fped[1:9000, ] %>% 
  mutate(tend = rep(seq(min(fstimes_surv$surv[, 1]), max(fstimes_surv$surv[, 1]), 
                        length.out = 1000L), 9),
         timenumeric = rep(c(4, 6, 8, 11, 13, 15, 18, 20, 23), each = 1000L), 
         .x = 0, .y = 0, 
         offset = 0,
         BoroughName = fped$BoroughName[1], 
         PropertyType = fped$PropertyType[1],
         method = "PAM",
         obs = rep(1:9, each = 1000)) 
test_grid_surv_pam$pred <- predict(pam, test_grid_surv_pam, type = "response")
test_grid_surv_pam <- test_grid_surv_pam %>%
  select(method, timenumeric, pred, obs, tend) %>%
  group_by(obs) %>% mutate(t_diff = tend - dplyr::lag(tend, 1, default = 0)) %>%
  mutate(sp = exp(-cumsum(.data$pred * .data$t_diff))) %>%
  select(method, timenumeric, sp, obs)

test_grid_surv_joined <- rbind(test_grid_surv, test_grid_surv_pam) %>%
  mutate(followup = rep(rep(seq(min(fstimes_surv$surv[, 1]), 
                                max(fstimes_surv$surv[, 1]), 
                                length.out = 1000L), 9), 2)) %>%
  mutate(method = factor(method, levels = c("PAM", "DRIFT")))
obsnames <- c("1" = "4 AM", "2" = "6 AM", "3" = "8 AM", "4" = "11 AM", "5" = "1 PM", 
              "6" = "3 PM", "7" = "6 PM", "8" = "8 PM", "9" = "11 PM")
obsnames2 <- c("1" = "4", "2" = "6", "3" = "8", "4" = "11", "5" = "13", 
               "6" = "15", "7" = "18", "8" = "21", "9" = "23")
test_grid_surv_joined$daytime <- as.numeric(obsnames2[test_grid_surv_joined$obs]) + 0.0

pq <- ggplot(test_grid_surv_joined %>% 
               filter(daytime %in% c(4, 13, 15, 21)),
             aes(followup, sp)) + 
  geom_line(aes(col = factor(obs))) + facet_wrap(method ~ .) + 
  theme_bw() +
  xlab("follow-up time (seconds)") +
  ylab("S(t)") +
  scale_y_continuous(breaks = seq(0, 1, by = .5)) +
  theme(legend.position =  "right", text = element_text(size = 15.5), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1)) 
pq

saveRDS(test_grid_surv_joined, "results/plotframe.Rds")

