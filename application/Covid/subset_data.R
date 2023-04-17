# Download data from:
# https://storage.googleapis.com/covid19-open-data/v3/aggregated.csv.gz
# into "data/"

# Read and subset data
library(data.table)
library(dplyr)

df <- fread(file="data/aggregated.csv")
df <- df %>% select(date, country_code, aggregation_level,
                    new_confirmed, population, 
              latitude, longitude,
              average_temperature_celsius, relative_humidity) %>% 
  filter(aggregation_level == 2, country_code == "US") %>% 
  filter(!is.na(latitude) & !is.na(longitude) & 
           !is.na(new_confirmed) & 
           !is.na(average_temperature_celsius) & 
           !is.na(relative_humidity)) %>% 
  mutate(date = as.numeric(as.Date(date) - as.Date("2020-01-01")),
         prevalence = new_confirmed / population)

# some final preparations

df <- df %>% filter(new_confirmed >= 0 & date <= 863)
df <- df %>% 
  mutate(
    prevalence = log(prevalence*1000 + 1),
    population = log(population)) %>%
  rename(
    temp = average_temperature_celsius,
    humid = relative_humidity
  ) %>% 
  select(date, population, latitude, longitude, temp, humid)

fwrite(df, "data/analysis_subset.csv")
