# Create plot
library(ggplot2)
library(ggmap)
library(mapproj)
library(viridis)
library(dplyr)
library(tidyr)

if(!file.exists("plot_data.RDS")){
  
  pldata <- lapply(list.files(pattern = "plotdata.*\\.RDS"), readRDS)
  names_eff <- c("days since start", "population", "latitude", "longitude", "temperature", "humidity")
  names_short <- c("date", "pop", "latlong", "temp", "humid")
  
  data <- as.data.frame(lapply(pldata[[3]], "[[", "value"))
  colnames(data) <- names_eff
  
  plotdata_nam <- as.data.frame(pldata[[1]][,-6])
  colnames(plotdata_nam) <- names_short # paste0("NAM_", names_short)
  plotdata_neat <- as.data.frame(pldata[[2]][,2:6])
  colnames(plotdata_neat) <- names_short # paste0("NEAT_", names_short)
  plotdata_str <- as.data.frame(lapply(pldata[[3]], function(x) (x$design_mat%*%x$coef)[,1]))
  colnames(plotdata_str) <- names_short # paste0("STR_", names_short)
  
  plotdata <- cbind(data, cbind(model=rep(c("NAM", "NEAT", "STR"), each = nrow(data)),
                                rbind(plotdata_nam, plotdata_neat, plotdata_str)))
  saveRDS(plotdata, file="plot_data.RDS")

}else{

  plotdata <- readRDS("plot_data.RDS")
  
}
  

plotdata_uni <- do.call("rbind",
                        list(
                          plotdata %>% select(`days since start`, date, model) %>% 
                            rename(pe = date, x = `days since start`) %>% mutate(var = "days since start"),
                          plotdata %>% select(population, pop, model) %>% 
                            rename(pe = pop, x = population) %>% mutate(var = "population"),
                          plotdata %>% select(temperature, temp, model) %>% 
                            rename(pe = temp, x = temperature) %>% mutate(var = "temperature"),
                          plotdata %>% select(humidity, humid, model) %>% 
                            rename(pe = humid, x = humidity) %>% mutate(var = "humidity")
                        ))

gg1 <- ggplot(plotdata_uni %>%
         mutate(model = factor(model, levels = unique(model),
                               labels = c("NAM", "DRAFT", "GAM"))) %>% 
         group_by(model, var) %>%
         mutate(pe = pe-mean(pe)) %>%
         ungroup %>% sample_n(1000000),
       aes(x = x, y = pe, colour = model)) +
  geom_line(linewidth=1.2) + facet_wrap(~var, scale="free", ncol = 2) +
  theme_bw() + xlab("Value") +
  ylab("Partial Effect") + theme(text = element_text(size = 14),
                                 legend.title = element_blank()) +
  scale_colour_manual(values = c("#009E73", "#999999", "#CC79A7")) +
  theme(legend.position="bottom")

ggsave(filename = "uni_effect.pdf", width = 6, height = 5)


# spatial
myLocation<-c(-130,
              23,
              -60,
              50)
myMap <- get_map(location = myLocation,
                 source="google",
                 maptype="terrain",
                 color = "bw",
                 crop=TRUE)

# state_df <- map_data("state", projection = "albers", parameters = c(39, 45))
state_df <- map_data("state")
county_df <- map_data("county")

ggplot(county_df, aes(long, lat, group = group)) +
  geom_polygon(colour = alpha("black", 1 / 2), size = 0.2) +
  geom_polygon(data = state_df, colour = "black", fill = NA) +
  theme_minimal() +
  theme(axis.line = element_blank(), axis.text = element_blank(),
        axis.ticks = element_blank(), axis.title = element_blank())

pd <- plotdata %>%
  # filter(model != "UNCONS") %>%
  filter(longitude >= -130 & longitude <= -60 &
           latitude >= 23 & latitude <= 50) %>%
  group_by(model) %>%
  mutate(latlong = latlong - mean(latlong)) %>%
  ungroup

ggmap(myMap) + # xlab("Longitude") + ylab("Latitude") +
  geom_point(data = pd %>% sample_n(1000000), #%>%
               # mutate(model = factor(model, levels = c("ONO", "PHOGAM", "NAM", "PHONAM"))),
             aes(x = longitude, y = latitude,
                 colour = latlong), alpha = 0.025, size = 8) +
  geom_polygon(data = state_df %>% arrange(-order),
               mapping = aes(x = long, y = lat, group = group),
               colour = "white", fill = NA, size = 0.2, alpha = 0.5) +
  geom_polygon(data = county_df %>% arrange(order),
               # aes(fill = latlong),
               mapping = aes(x = long, y = lat, group = group),
               colour = alpha("white", 1 / 2), size = 0.06, alpha=0.005) +
  scale_colour_viridis_c(option = 'magma', direction = -1,
                         name = "") +
  guides(alpha = "none", size = "none") + # ggtitle("Geographic Location Effect") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size=14),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position = "bottom",
        legend.key.width=unit(1.2,"cm")
        ) +
  facet_wrap(~ model, ncol = 2)

ggsave(file = "spatial.jpeg", device = "jpeg"#,
       #width = 7, height = 5.5
       )

effect <- readRDS("plotspatial_neat.RDS")[,4]
county_df$eff <- effect - mean(effect)

state_df_proj <- map_data("state", projection = "albers", parameters = c(39, 45))
state_df <- state_df %>% left_join(state_df_proj, by = c("group", "order"))
county_df_proj <- map_data("county", projection = "albers", parameters = c(39, 45))
county_df <- county_df %>% left_join(county_df_proj, by = c("group", "order"))

gg2 <- ggplot(county_df, aes(long.y, lat.y, group = group)) +
  geom_polygon(aes(fill = eff), colour = alpha("white", 0.1), size = 0.12) +
  geom_polygon(data = state_df, colour = "white", fill = NA) +
  coord_fixed() +
  theme_minimal() +
  theme(axis.line = element_blank(), axis.text = element_blank(),
        axis.ticks = element_blank(), axis.title = element_blank()) +
  scale_fill_viridis_c(option = 'mako', direction = -1,
                         name = "") +
  theme(legend.position = "bottom",
        legend.key.width=unit(1.6,"cm"),
        text = element_text(size = 16))

gg2 + theme(plot.margin = unit(rep(1, 4), "cm"))

ggsave(file = "spatial_draft.pdf"#, device = "jpeg"#,
       #width = 7, height = 5.5
)

g1 <- gridExtra::grid.arrange(gg1, gg2, ncol=2, widths=c(1.3,1))
ggsave(file = "covid_matrix_fig.pdf", g1, width = 8, height = 4)
