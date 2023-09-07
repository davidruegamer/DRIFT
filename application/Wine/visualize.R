
library("tidyverse")

odir <- "Results"

pdat <- readRDS(file.path(odir, "pdat.rds"))
plr <- readRDS(file.path(odir, "plr.rds"))
perf <- readRDS(file.path(odir, "perf.rds"))

col2 <- colorspace::diverge_hcl(2)

ggplot(pdat, aes(x = x, y = bhat, linetype = split)) +
  geom_line(alpha = 0.6, aes(color = "DRIFT")) +
  geom_line(alpha = 0.3, data = plr, aes(color = "POLR")) +
  facet_wrap(~ pred, labeller = as_labeller(
    c("x.V1" = "fixed acidity", "x.V2" = "volatile acidity", "x.V3" = "citric acid", 
      "x.V4" = "residual sugar", "x.V5" = "chlorides")), nrow = 2) +
  theme_bw() +
  labs(y = "partial effect", x = "value", color = element_blank()) +
  theme(text = element_text(size = 13.5), legend.position = "top",
        axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1)) +
  guides(linetype = "none") +
  scale_color_manual(values = c("DRIFT" = col2[1], "POLR" = col2[2]))

ggsave(file.path(odir, "wine.pdf"), width = 6, height = 5.5)

