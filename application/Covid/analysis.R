library(dplyr)
library(deepregression)
library(deeptrafo)
devtools::load_all("../../neat/R/")

# Choose size
size <- "small"
# size <- "large"
save_model <- FALSE

# Read data
df <- read.csv("data/analysis_subset.csv")
df <- df %>% 
  rename(
    temp = average_temperature_celsius,
    humid = relative_humidity
  ) %>% 
  mutate(
    cases = new_confirmed * 1.0
  ) %>% 
  filter(
    cases >=0
  )
# Train/Test
test <- df %>% filter(date > 700)
df <- df %>% filter(date <= 700)

if(size == "small")
{
  
  df <- df[sample(1:nrow(df), 5000),]
  bs <- 32L
  eps <- 2000L
  
}else{
  
  bs <- 1024L
  eps <- 150L

}
ps <- 15L


# Define deep networks
deep_mod <- function(x) x %>% 
  layer_dense(units = 100, activation = "relu", use_bias = FALSE) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 100, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)

deep_mod_nam <- function(x) x %>% 
  layer_dense(units = 20, activation = "tanh", use_bias = FALSE) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 20, activation = "tanh") %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)

feature_net <- function(x) x %>% 
  layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
  layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

feature_net_bi <- function(x){ 
  
  base1 <- tf_stride_cols(x, 1) %>% 
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 32, activation = "relu") %>% 
    layer_dense(units = 5)
  
  base2 <- tf_stride_cols(x, 2) %>% 
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 32, activation = "relu") %>% 
    layer_dense(units = 5)
  
  tf_row_tensor(base1, base2) %>% 
    layer_dense(1)
  
}

# Define model formulas
form_structured <- ~ 1 + s(date) + s(population) + te(latitude, longitude) + 
  s(temp) + s(humid) 
form_nam <- ~ 1 + snam(date) + snam(population) + tenam(latitude, longitude) + 
  snam(temp) + snam(humid) 
form_sls <- cases ~ 1 + s(date) + s(population) + te(latitude, longitude) + 
  s(temp) + s(humid) 
form_neat <- cases  ~ 1 + snam(date) + snam(population) + tenam(latitude, longitude) + 
  snam(temp) + snam(humid) 

# Init and fit models
mod_str <- deepregression(
  y = df$cases,
  list_of_formulas = list(form_structured),
  data = df,
  family = "poisson",
  list_of_deep_models = NULL,
  orthog_options = orthog_control(orthogonalize = TRUE)
)

if(file.exists("weights_str.hdf5")){
  
  mod_str$model$load_weights(filepath="weights_str.hdf5", by_name = FALSE)
  
}else{
  
  hist_str <- mod_str %>% fit(epochs = eps, early_stopping = TRUE, 
                          patience = ps, batch_size = bs)
  
  save_model_weights_hdf5(mod_str$model, filepath="weights_str.hdf5")
  
}

Metrics::rmse(test$cases, mod_str %>% predict(test))

mod_nam <- deepregression(
  y = df$cases,
  list_of_formulas = list(form_nam, ~1),
  data = df,
  list_of_deep_models = 
    list(
      snam = feature_net,
      tenam = feature_net_bi
    ),
  orthog_options = orthog_control(orthogonalize = FALSE)
)

if(file.exists("weights_nam.hdf5")){
  
  mod_nam$model$load_weights(filepath="weights_nam.hdf5", by_name = FALSE)
  
}else{
  
  hist_unc <- mod_nam %>% fit(epochs = eps, early_stopping = TRUE, 
                                    patience = ps, batch_size = bs)
  
  if(save_model)
    save_model_weights_hdf5(mod_nam$model, filepath="weights_nam.hdf5")
  
}

Metrics::rmse(test$new_confirmed, mod_nam %>% predict(test))
# 0.7142373

mod_sls <- deeptrafo(form_sls, 
                     response_type = "count",
                     data = df,
                     list_of_deep_models = NULL,
                     orthog_options = orthog_control(orthogonalize = FALSE)
)

if(file.exists("weights_sls.hdf5")){
  
  mod_sls$model$load_weights(filepath="weights_sls.hdf5", by_name = FALSE)
  
}else{
  
  hist_sls <- mod_sls %>% fit(epochs = eps, early_stopping = TRUE, 
                              patience = ps, batch_size = bs)
  
  if(save_model)
    save_model_weights_hdf5(mod_sls$model, filepath="weights_sls.hdf5")
  
}

Metrics::rmse(test$new_confirmed, mod_sls %>% predict(test))
# 0.6670314


neat <- deeptrafo(form_neat, 
                      response_type = "count",
                      data = df,
                      list_of_deep_models = list(
                        dnn = deep_mod_nam,
                        snam = feature_net,
                        tenam = feature_net_bi
                      ),
                      orthog_options = orthog_control(orthogonalize = FALSE)
)

if(file.exists("weights_nam.hdf5")){
  
  neat$model$load_weights(filepath="weights_nam.hdf5", by_name = FALSE)
  
}else{
  
  hist_nam <- neat %>% fit(epochs = eps, early_stopping = TRUE, 
                          patience = ps, batch_size = bs)
  
  if(save_model)
    save_model_weights_hdf5(neat$model, filepath="weights_nam.hdf5")
   
}

Metrics::rmse(test$new_confirmed, neat %>% predict(test))
# 0.6670314


# # Create plot
# library(ggplot2)
# library(ggmap)
# library(mapproj)
# library(viridis)
# 
# plotdata <- readRDS("plotdata.RDS")
# 
# plotdata_uni <- do.call("rbind", 
#                         list(
#                           plotdata %>% select(date, `s(date)`, method) %>% rename(pe = `s(date)`,
#                                                                           x = date) %>% mutate(var = "date"), 
#                           plotdata %>% select(population, `s(population)`, method) %>% rename(pe = `s(population)`,
#                                                                                       x = population) %>% mutate(var = "population"),
#                           plotdata %>% select(temp, `s(temp)`, method) %>% rename(pe = `s(temp)`,
#                                                                           x = temp) %>% mutate(var = "temp"),
#                           plotdata %>% select(humid, `s(humid)`, method) %>% rename(pe = `s(humid)`,
#                                                                             x = humid) %>% mutate(var = "humid")
#                         ))
# 
# plotdata_uni$var[plotdata_uni$var=="humid"] <- "humidity"
# plotdata_uni$var[plotdata_uni$var=="temp"] <- "temperature"
# plotdata_uni$var[plotdata_uni$var=="date"] <- "days since start"
# 
# ggplot(plotdata_uni %>% filter(method != "UNCONS") %>% 
#          group_by(method, var) %>% 
#          mutate(pe = pe - mean(pe)) %>% 
#          ungroup, 
#        aes(x = x, y = pe, colour = method)) + 
#   geom_line(size=1.2) + facet_wrap(~var, scale="free", ncol = 2) + 
#   theme_bw() + xlab("Value") + 
#   ylab("Partial Effect") + theme(text = element_text(size = 14),
#                                  legend.title = element_blank()) + 
#   scale_colour_manual(values = c("#009E73", "#E69F00", "#999999", "#CC79A7", "#56B4E9")) + 
#   theme(legend.position="bottom")
# 
# ggsave(filename = "uni_effect.pdf", width = 6, height = 5)
# 
# 
# # spatial
# myLocation<-c(-130,  
#               23, 
#               -60, 
#               50)
# myMap <- get_map(location = myLocation, 
#                  source="google", 
#                  maptype="terrain", 
#                  color = "bw",
#                  crop=TRUE)
# 
# # state_df <- map_data("state", projection = "albers", parameters = c(39, 45))
# state_df <- map_data("state")
# county_df <- map_data("county")
# 
# ggplot(county_df, aes(long, lat, group = group)) +
#   geom_polygon(colour = alpha("black", 1 / 2), size = 0.2) + 
#   geom_polygon(data = state_df, colour = "black", fill = NA) +
#   theme_minimal() + 
#   theme(axis.line = element_blank(), axis.text = element_blank(),
#         axis.ticks = element_blank(), axis.title = element_blank())
# 
# pd <- plotdata %>% 
#   # filter(method != "UNCONS") %>% 
#   filter(longitude >= -130 & longitude <= -60 & 
#            latitude >= 23 & latitude <= 50) %>% 
#   group_by(method) %>% 
#   mutate(`te(latitude, longitude)` = `te(latitude, longitude)` - mean(`te(latitude, longitude)`)) %>% 
#   ungroup 
# 
# ggmap(myMap) + # xlab("Longitude") + ylab("Latitude") + 
#   geom_point(data = pd %>% filter(!method %in% c("UNCONS", "PHO")) %>% 
#                mutate(method = factor(method, levels = c("ONO", "PHOGAM", "NAM", "PHONAM"))), 
#              aes(x = longitude, y = latitude, 
#                  colour = `te(latitude, longitude)`), alpha = 0.025, size = 8) + 
#   geom_polygon(data = state_df %>% arrange(-order),
#                mapping = aes(x = long, y = lat, group = group),
#                colour = "white", fill = NA, size = 0.2, alpha = 0.5) +
#   geom_polygon(data = county_df %>% arrange(order),
#                # aes(fill = `te(latitude, longitude)`),
#                mapping = aes(x = long, y = lat, group = group),
#                colour = alpha("white", 1 / 2), size = 0.06, alpha=0.005) +
#   scale_colour_viridis_c(option = 'magma', direction = -1, 
#                          name = "") + 
#   guides(alpha = "none", size = "none") + # ggtitle("Geographic Location Effect") +
#   theme(plot.title = element_text(hjust = 0.5),
#         text = element_text(size=14),
#         axis.title.x=element_blank(),
#         axis.text.x=element_blank(),
#         axis.ticks.x=element_blank(),
#         axis.title.y=element_blank(),
#         axis.text.y=element_blank(),
#         axis.ticks.y=element_blank(),
#         legend.position = "bottom",
#         legend.key.width=unit(1.2,"cm")
#         ) + 
#   facet_wrap(~ method, ncol = 2) 
# 
# ggsave(file = "spatial.jpeg", device = "jpeg"#,
#        #width = 7, height = 5.5
#        )
# 
# library(mgcv)
# tetr <- smoothCon(te(latitude, longitude), data = df %>% select(latitude, longitude))[[1]]
# Xp <- PredictMat(tetr, county_df %>% select(lat, long) %>% 
#                    rename(latitude = lat, longitude = long))
# coef_pho <- readRDS("coef_pho.RDS")
# effect <- Xp%*%coef_pho[grepl("latitude", names(coef_pho))]
# county_df$eff <- effect[,1] - mean(effect[,1])
# 
# state_df_proj <- map_data("state", projection = "albers", parameters = c(39, 45))
# state_df <- state_df %>% left_join(state_df_proj, by = c("group", "order"))
# county_df_proj <- map_data("county", projection = "albers", parameters = c(39, 45))
# county_df <- county_df %>% left_join(county_df_proj, by = c("group", "order"))
# 
# ggplot(county_df, aes(long.y, lat.y, group = group)) + 
#   geom_polygon(aes(fill = eff), colour = alpha("white", 0.1), size = 0.12) +
#   geom_polygon(data = state_df, colour = "white", fill = NA) +
#   coord_fixed() +
#   theme_minimal() +
#   theme(axis.line = element_blank(), axis.text = element_blank(),
#         axis.ticks = element_blank(), axis.title = element_blank()) +
#   scale_fill_viridis_c(option = 'magma', direction = -1, 
#                          name = "") + 
#   theme(legend.position = "bottom",
#         legend.key.width=unit(1.6,"cm"),
#         text = element_text(size = 16))
# 
# ggsave(file = "spatial_phogam.pdf"#, device = "jpeg"#,
#        #width = 7, height = 5.5
# )
