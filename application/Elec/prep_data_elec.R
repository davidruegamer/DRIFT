p <- file.path(getwd(), "application","ts_example")
source(file.path(p, "utils.R"))

parser <- ArgumentParser()
parser$add_argument("-l", "--lags", default=48,
                    help="Number of consecutive lags starting from 1 going up to...")
args <- parser$parse_args()

prep_data <- function(first_date = "2014-07-01 00:00:00",
                      last_date = "2014-07-07 23:00:00",
                      l_ags = NULL){
  
  ## "electricity": hourly electricity consumption (kwh) for 370 households
  
  d <- fread(file.path(p, "elec.csv"))
  d[, ds := as.POSIXct(V1, format = "%Y-%m-%d %H:%M:%S", tz = tzz)]; d$V1 <- NULL
  
  # create features and prepare for cluster
  d_sets <- equip_d(create_lags(d, lags = 1:as.numeric(l_ags)))

  # get train test split
  next_day <- as.POSIXct(last_date, format = "%Y-%m-%d %H:%M:%S") + 24*60*60
  nextnext_day <- next_day + 24*60*60
  
  d_val_tr <- d_sets %>% filter(t_ime >= first_date & t_ime <= last_date)
  d_val_tst <- d_sets %>% filter(t_ime > last_date & t_ime <= next_day)
  d_tst_tr <- d_sets %>% filter(t_ime >= first_date & t_ime <= next_day)
  d_tst_tst <- d_sets %>% filter(t_ime > next_day & t_ime <= nextnext_day)

  rm(d_sets); gc()
  
  return(list(d_val_tr = d_val_tr,
              d_val_tst = d_val_tst,
              d_tst_tr = d_tst_tr,
              d_tst_tst = d_tst_tst))
  
}

d <- prep_data(l_ags = args$lags)
saveRDS(d, file = file.path(p, "electricity.RDS"))