
library("tidyverse")
library("ggpubr")
theme_set(theme_bw() + theme(text = element_text(size = 13.5),
                             legend.position = "top", legend.justification = "right"))

resids <- readRDS("martingales.Rds") |>
  mutate(split = factor(split))

res <- readRDS("res.Rds") |>
  mutate(split = factor(split))

pdat <- readRDS("plotframe.Rds")

p1 <- ggplot(resids |> filter(split == 1), aes(x = ours, y = pam, group = split)) +
  geom_point(alpha = 0.1, show.legend = FALSE) +
  # geom_smooth(method = "lm", show.legend = FALSE) +
  geom_abline(slope = 1, intercept = 0, linetype = 2) +
  labs(x = "test residuals DRIFT", y = "test residuals PAM")

p2 <- ggplot(res, aes(x = method, y = IBS)) +
  geom_boxplot() +
  facet_wrap(~ Q, scales = "free", labeller = as_labeller(
    c("Q25" = "25th percentile", "Q50" = "50th percentile",
      "Q75" = "75th percentile")
  )) +
  labs(x = element_blank(), y = "integrated Brier score") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1))

p3 <- ggplot(pdat |> filter(daytime %in% c(4, 13, 23)),
             aes(x = followup, y = log(-log(sp)), color = ordered(daytime),
                 linetype = method)) +
  geom_line() +
  scale_linetype_manual(values = c("DRIFT" = 1, "PAM" = 4)) +
  scale_color_viridis_d(end = 0.6, option = "B") +
  labs(x = "follow-up time (seconds)", y = "log cumulative hazards",
       color = "daytime (hours)", method = element_blank()) +
  coord_cartesian(ylim = c(-5, 2.5))

ggarrange(p2, p3, p1, common.legend = TRUE, widths = c(2, 1, 1), nrow = 1)
ggsave("surv.pdf", width = 11.5, height = 3)
