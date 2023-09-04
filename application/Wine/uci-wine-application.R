# DRIFT: POLR NAM for the wine data
# May 2023

# Dependencies ------------------------------------------------------------

library("deeptrafo")
library("tram")
library("tidyverse")

odir <- "Results"

if (!dir.exists(odir))
  dir.create(odir, recursive = TRUE)

# Helpers -----------------------------------------------------------------

u_scale <- function (col, min_col, max_col, eps = 0.0) {
  ((col - min_col) / (eps + max_col - min_col))
}

col_scale <- function(col_train,
                      spatz = 0.05,
                      col_test,
                      eps = 0.0) {
  # use only train to determine scale-parameter
  max_col <- max(col_train) * (1 + spatz)
  min_col <- min(col_train) * (1 - spatz)
  col_train <- u_scale(col_train, min_col = min_col, max_col = max_col, eps)
  col_test <- u_scale(col_test, min_col = min_col, max_col = max_col, eps)
  return (list(col_train = col_train, col_test = col_test))
}

load_data <- function(path, split_num = 0, spatz = 0.05, x_scale = TRUE, eps = 0) {
  idx_train <- read.table(paste0(path, 'index_train_', split_num, '.txt')) + 1 
  idx_test <- read.table(paste0(path, 'index_test_', split_num, '.txt')) + 1 
  y_col <- read.table(paste0(path, 'index_target.txt'))  + 1
  x_cols <- read.table(paste0(path, 'index_features.txt'))  + 1
  runs <- as.numeric(read.table(paste0(path, 'n_splits.txt')))
  dat <- as.matrix(read.table(paste0(path, 'data.txt')))
  X <- dat[, x_cols$V1]
  y <- dat[, y_col$V1]
  X_train <- X[idx_train$V1, ]
  y_train <- y[idx_train$V1]
  X_test <- X[idx_test$V1, ]
  y_test <- y[idx_test$V1]
  max_y <- max(y_train) * (1 + spatz)
  min_y <- min(y_train) * (1 - spatz)
  scale <- max_y - min_y
  
  if (x_scale == TRUE) {
    X_train <- as.matrix(X_train) # also in case of 1 x
    for (i in 1:ncol(X_train)) {
      X_s2 <- col_scale(col_train = X_train[, i], spatz = 0, col_test = X_test[, i], eps)
      X_train[, i] <- X_s2$col_train
      X_test[, i] <- X_s2$col_test
    }

  }
  list(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
       runs = runs, scale = scale)
}

get_data_wine <- function(path, split_num = 0, spatz = 0.05, x_scale = FALSE) {
  name <- 'wine-quality-red'
  ret <- load_data(path, split_num, spatz, x_scale)
  ret$name <- name
  ret
}

preproc <- function(vec, labs = 1:6, levs = 3:8, retfac = FALSE) {
  tmp <- factor(vec, levels = levs, labels = labs, ordered = TRUE)
  mat <- model.matrix( ~ 0 + tmp, contrasts.arg = list(tmp = "contr.treatment"))
  if (retfac)
    return(tmp)
  mat
}

dat_to_df <- function(split_num = 0) {
  dat <- get_data_wine("Data/", x_scale = TRUE, split_num = split_num)
  dat$y_trainc <- preproc(dat$y_train, retfac = TRUE)
  dat$y_testc <- preproc(dat$y_test, retfac = TRUE)
  ydtr <- preproc(dat$y_train, retfac = FALSE)
  ydte <- preproc(dat$y_test, retfac = FALSE)
  dtrain <- data.frame(y = dat$y_trainc, x = dat$X_train)
  dtest <- data.frame(y = dat$y_testc, x = dat$X_test)
  return(list(train = dtrain, test = dtest,
              y_dummy_train = ydtr,
              y_dummy_test = ydte))
}

nn <- function()  keras_model_sequential() %>% 
  layer_dense(input_shape = 1L, units = 8L, activation = "relu") %>% 
  layer_dense(units = 8L, activation = "relu") %>% 
  layer_dense(units = 8L, activation = "relu") %>% 
  layer_dense(units = 8L, activation = "relu") %>% 
  layer_dense(units = 1L, activation = "linear")

# Analysis ----------------------------------------------------------------

npreds <- 5

out <- lapply(0:19, \(split) { 
  dat <- dat_to_df(split_num = split)
  m <- Polr(y ~ x.V1 + x.V2 + x.V3 + x.V4 + x.V5, data = dat$train)
  lodm <- lapply(1:npreds, \(x) nn())
  names(lodm) <- paste0("nn", 1:npreds)
  mm <- PolrNN(y ~ 0 + nn1(x.V1) + nn2(x.V2) + nn3(x.V3) + nn4(x.V4) + nn5(x.V5),
               data = dat$train, list_of_deep_models = lodm,
               optimizer = optimizer_adam(learning_rate = 1e-3, decay = 1e-4))
  fit(mm, epochs = 2e2, validation_split = NULL)
  polr_train <- - logLik(m)/nrow(dat$train)
  polr_test <- - logLik(m, newdata = dat$test)/nrow(dat$test)
  nam_train <- logLik(mm, convert_fun = mean)
  nam_test <- logLik(mm, newdata = dat$test, convert_fun = mean)
  preds <- do.call("rbind", lapply(seq_len(npreds), \(pred) {
    nd <- matrix(seq(min(dat$train[, 1 + pred]), max(dat$train[, 1 + pred]), length.out = 1e3))
    pp <- lodm[[pred]](nd)$numpy()
    data.frame(pred = colnames(dat$train)[1 + pred], x = nd, bhat = pp)
  }))
  perf <- data.frame(
    mod = rep(c("polr", "nam"), each = 2),
    set = rep(c("train", "test"),  2),
    nll = c(polr_train, polr_test, nam_train, nam_test)
  )
  cfx <- coef(m)
  list(preds = preds, perf = perf, polr = cfx)
})

pdat <- bind_rows(lapply(out, `[[`, 1), .id = "split") %>% 
  group_by(split, pred) %>% mutate(bhat = bhat - mean(bhat)) %>% 
  filter(pred != "x.V5")
perf <- bind_rows(lapply(out, `[[`, 2), .id = "split")
sx <- seq(0, 1, length.out = 1e3)
plr <- bind_rows(lapply(out, `[[`, 3), .id = "split") %>% 
  pivot_longer(names_to = "pred", values_to = "coef", x.V1:x.V4) %>% 
  group_by(split, pred) %>% mutate(bhat = list(data.frame(
    x = sx, bhat = - coef * sx - mean(-coef * sx)))) %>% 
  unnest(bhat)

saveRDS(pdat, file.path(odir, "pdat.rds"))
saveRDS(perf, file.path(odir, "perf.rds"))

col2 <- colorspace::diverge_hcl(2)

ggplot(pdat, aes(x = x, y = bhat, linetype = split)) +
  geom_line(alpha = 0.6, aes(color = "DRIFT")) +
  geom_line(alpha = 0.3, data = plr, aes(color = "POLR")) +
  facet_wrap(~ pred, labeller = as_labeller(
    c("x.V1" = "fixed acidity", "x.V2" = "volatile acidity", "x.V3" = "citric acid", 
      "x.V4" = "residual sugar", "x.V5" = "chlorides")), nrow = 2) +
  theme_bw() +
  labs(y = "partial effect of x", x = "x", color = element_blank()) +
  theme(text = element_text(size = 13.5), legend.position = "top",
        axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1)) +
  guides(linetype = "none") +
  scale_color_manual(values = c("DRIFT" = col2[1], "POLR" = col2[2]))

ggsave(file.path(odir, "wine.pdf"), width = 5, height = 5.5)

rres <- perf %>% 
  group_by(mod, set) %>% 
  summarize(mean = mean(nll), sd = sd(nll))
rres

saveRDS(rres, file.path(odir, "rres.rds"))

