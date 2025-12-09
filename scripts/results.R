# Packages
library(tidyverse)
library(data.table)
library(lubridate)
library(fixest)
library(modelsummary)
library(broom)
library(ggplot2)
library(stringr)


# Paths
data_dir <- "datasets"
out_tab  <- "tables"
out_fig  <- "figures"

# Helper functions

read_chat_trends <- function(path) {
  raw <- readLines(path, warn = FALSE, encoding = "UTF-8")
  keep_idx <- which(str_detect(raw, "^Semana,") |
                      str_detect(raw, "^\\d{4}-\\d{2}-\\d{2},"))
  raw2 <- raw[keep_idx]
  
  df <- read_csv(I(paste(raw2, collapse = "\n")),
                 col_types = cols(.default = col_character()))
    names(df) <- c("week", "chat_interest")
  
  df %>%
    mutate(
      week = as.Date(week),
      chat_interest = as.numeric(chat_interest)
    ) %>%
    arrange(week)
}

# Month difference helper for event time
month_diff <- function(d, ref) {
  (year(d) - year(ref)) * 12 + (month(d) - month(ref))
}

# Save table to html + tex 
save_models <- function(models, file_stub, title = NULL) {
  # HTML
  modelsummary(models,
               stars = TRUE,
               output = file.path(out_tab, paste0(file_stub, ".html")),
               title = title)
  
  # LaTeX
  modelsummary(models,
               stars = TRUE,
               output = file.path(out_tab, paste0(file_stub, ".tex")),
               title = title)
}

# Read data

# Trends
trends_path <- file.path(data_dir, "chat google trends.csv")
chat_trends <- read_chat_trends(trends_path)

# Part I main estimation sample (sold-only, Qwen prices)
chat_panel_path <- file.path(data_dir, "chat_sales_panel_qwen_prices.csv")
chat_sales <- fread(chat_panel_path) %>% as_tibble()

# Part II main estimation sample (sold-only, Qwen prices)
cross_path <- file.path(data_dir, "multi_keyword_cross_section_qwen_prices.csv")
cross_sold <- fread(cross_path) %>% as_tibble()

# Part I
chat_sales <- chat_sales %>%
  mutate(
    date = as.Date(date),
    post = as.integer(post),
    sold = as.integer(sold),
    # ensure distance used is numeric
    distance_used = as.numeric(distance_used),
    embedding_distance = as.numeric(embedding_distance),
    attention_index = as.numeric(attention_index),
    is_com = as.integer(is_com),
    length = as.numeric(length),
    age_years = as.numeric(age_years),
    has_hyphen = as.integer(has_hyphen),
    has_digit = as.integer(has_digit),
    exact_match = as.integer(exact_match),
    log_price_obs = as.numeric(log_price_obs),
    price_obs = as.numeric(price_obs)
  ) %>%
  filter(sold == 1)

# Part II
cross_sold <- cross_sold %>%
  mutate(
    sold = as.integer(sold),
    distance_used = as.numeric(distance_used),
    embedding_distance = as.numeric(embedding_distance),
    S_k = as.numeric(S_k),
    is_com = as.integer(is_com),
    length = as.numeric(length),
    age_years = as.numeric(age_years),
    has_hyphen = as.integer(has_hyphen),
    has_digit = as.integer(has_digit),
    token_count = as.numeric(token_count),
    exact_match = as.integer(exact_match),
    log_price_obs = as.numeric(log_price_obs),
    price_obs = as.numeric(price_obs),
    cluster_keyword = as.factor(cluster_keyword),
    tld = as.factor(tld)
  ) %>%
  filter(sold == 1)


# PART I — Main Results
# We estimate
# log P_it = alpha_i + lambda_t + beta1 d_i + beta2 Post_t + beta3 Post_t*d_i + eps
# With domain FE + time FE, identified term is Post x distance.

# Naive cross-sectional-ish OLS (no FE) for comparison
m1_naive <- feols(
  log_price_obs ~ distance_used + post + post:distance_used +
    length + age_years + is_com + has_hyphen + has_digit + exact_match,
  data = chat_sales,
  vcov = ~ domain
)

# Preferred FE DID slope-change model
m1_fe <- feols(
  log_price_obs ~ post:distance_used +
    length + age_years + is_com + has_hyphen + has_digit + exact_match |
    domain + date,
  data = chat_sales,
  vcov = ~ domain
)


# Save Table: Part I main
save_models(
  list("Naive OLS" = m1_naive, "Domain+Time FE" = m1_fe),
  "tab_part1_main",
  title = "Part I — Effect of ChatGPT Shock on the Price–Distance Gradient"
)

# PART I — Event-Study
# log P_it = alpha_i + lambda_t + sum_{k != -1} theta_k (d_i * 1{t=k}) + eps

# Shock date (ChatGPT-3.5 public release month)
shock_date <- as.Date("2022-11-01")

chat_sales_es <- chat_sales %>%
  mutate(
    event_time = month_diff(date, shock_date)
  )

es_min <- -24 # Adjustable
es_max <-  24

chat_sales_es <- chat_sales_es %>%
  filter(event_time >= es_min, event_time <= es_max)

m_es <- feols(
  log_price_obs ~ i(event_time, distance_used, ref = -1) +
    length + age_years + is_com + has_hyphen + has_digit + exact_match |
    domain + date,
  data = chat_sales_es,
  vcov = ~ domain
)

# Event-study plot
es_coefs <- broom::tidy(m_es) %>%
  filter(str_detect(term, "event_time::")) %>%
  mutate(
    k = as.integer(str_extract(term, "-?\\d+"))
  ) %>%
  arrange(k)

p_es <- ggplot(es_coefs, aes(x = k, y = estimate)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point() +
  geom_errorbar(aes(ymin = estimate - 1.96 * std.error,
                    ymax = estimate + 1.96 * std.error),
                width = 0.2) +
  labs(
    title = "Event-Study: Change in the Price–Distance Gradient Around ChatGPT",
    subtitle = "Coefficients are slopes on distance relative to month -1",
    x = "Months relative to Nov 2022",
    y = "Estimated slope shift"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_fig, "fig_part1_event_study.png"),
       p_es, width = 8, height = 4.8, dpi = 300)

# Save event-study model table
save_models(list("Event-study FE" = m_es), "tab_part1_event_study")

# PART I — Robustness (Appendix)

# Alternative bandwidths around shock
bandwidth_months <- list(
  "±6m"  = 6,
  "±12m" = 12,
  "±24m" = 24
)

band_models <- map(bandwidth_months, function(bw) {
  df <- chat_sales %>%
    mutate(event_time = month_diff(date, shock_date)) %>%
    filter(event_time >= -bw, event_time <= bw)
  
  feols(
    log_price_obs ~ post:distance_used +
      length + age_years + is_com + has_hyphen + has_digit + exact_match |
      domain + date,
    data = df,
    vcov = ~ domain
  )
})

names(band_models) <- names(bandwidth_months)
save_models(band_models, "tab_part1_bandwidths",
            title = "Appendix — Part I Robustness: Alternative Event Windows")

# Placebo shock dates (pre-2022)
placebo_dates <- as.Date(c("2021-06-01", "2021-11-01", "2022-03-01"))

placebo_models <- map(placebo_dates, function(pd) {
  df <- chat_sales %>%
    mutate(post_pl = as.integer(date >= pd))
  
  feols(
    log_price_obs ~ post_pl:distance_used +
      length + age_years + is_com + has_hyphen + has_digit + exact_match |
      domain + date,
    data = df,
    vcov = ~ domain
  )
})

names(placebo_models) <- paste0("Placebo ", format(placebo_dates, "%Y-%m"))
save_models(placebo_models, "tab_part1_placebos",
            title = "Appendix — Part I Robustness: Placebo Shock Dates")

# Functional form checks (distance transformations)
chat_sales <- chat_sales %>%
  mutate(
    dist_sq = distance_used^2,
    dist_log = log1p(distance_used) # safe even if near 0
  )

m_ff_lin <- m1_fe

m_ff_quad <- feols(
  log_price_obs ~ post:distance_used + post:dist_sq +
    length + age_years + is_com + has_hyphen + has_digit + exact_match |
    domain + date,
  data = chat_sales,
  vcov = ~ domain
)

m_ff_log <- feols(
  log_price_obs ~ post:dist_log +
    length + age_years + is_com + has_hyphen + has_digit + exact_match |
    domain + date,
  data = chat_sales,
  vcov = ~ domain
)

save_models(
  list("Linear" = m_ff_lin, "Quadratic" = m_ff_quad, "Log(1+d)" = m_ff_log),
  "tab_part1_functional_forms",
  title = "Appendix — Part I Robustness: Functional Form"
)

# PART II — Cross-Section Main Results
# log P_ik = gamma_k + delta_k d_ik + X' psi + xi

# Pooled model with category FE
m2_pooled <- feols(
  log_price_obs ~ distance_used +
    length + age_years + is_com + has_hyphen + has_digit +
    token_count + exact_match |
    cluster_keyword,
  data = cross_sold,
  vcov = "hetero"
)

# Category-by-category gradients (no FE needed inside cluster)
m2_by_cluster <- cross_sold %>%
  group_by(cluster_keyword) %>%
  group_map(~{
    feols(
      log_price_obs ~ distance_used +
        length + age_years + is_com + has_hyphen + has_digit +
        token_count + exact_match,
      data = .x,
      vcov = "hetero"
    )
  })

cluster_levels <- levels(cross_sold$cluster_keyword)
names(m2_by_cluster) <- paste0("Cluster: ", cluster_levels)

# Save pooled + selected cluster models
save_models(
  c(list("Pooled + Category FE" = m2_pooled),
    m2_by_cluster[seq_along(m2_by_cluster)]),
  "tab_part2_cluster_gradients",
  title = "Part II — Cross-Sectional Price–Distance Gradients Across Keywords"
)

# PART II — Popularity vs Gradient

# Extract estimated delta_k from per-cluster regressions
delta_df <- map2_dfr(m2_by_cluster, cluster_levels, function(mod, k) {
  tibble(
    cluster_keyword = k,
    delta_hat = coef(mod)[["distance_used"]],
    se = sqrt(diag(vcov(mod)))[["distance_used"]]
  )
})

# Merge unique S_k by cluster
Sk_df <- cross_sold %>%
  distinct(cluster_keyword, S_k)

delta_df <- delta_df %>%
  left_join(Sk_df, by = "cluster_keyword")

# Simple second-stage relationship
m2_pop <- feols(delta_hat ~ S_k, data = delta_df, vcov = "hetero")

save_models(
  list("Delta_k on popularity" = m2_pop),
  "tab_part2_popularity_slope",
  title = "Part II — Relationship Between Keyword Popularity and Gradient Steepness"
)

# Plot: S_k vs delta_hat
p_pop <- ggplot(delta_df, aes(x = S_k, y = delta_hat)) +
  geom_point() +
  geom_errorbar(aes(ymin = delta_hat - 1.96 * se,
                    ymax = delta_hat + 1.96 * se),
                width = 0) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(
    title = "Keyword Popularity and Price–Distance Gradient",
    x = "Keyword popularity (S_k)",
    y = "Estimated distance slope (delta_k)"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_fig, "fig_part2_popularity_gradient.png"),
       p_pop, width = 7, height = 4.8, dpi = 300)

# ART II — Appendix Robustness

# Functional forms (log-distance, quadratic)
cross_sold <- cross_sold %>%
  mutate(
    dist_sq = distance_used^2,
    dist_log = log1p(distance_used)
  )

m2_ff_lin <- m2_pooled

m2_ff_quad <- feols(
  log_price_obs ~ distance_used + dist_sq +
    length + age_years + is_com + has_hyphen + has_digit +
    token_count + exact_match |
    cluster_keyword,
  data = cross_sold, vcov = "hetero"
)

m2_ff_log <- feols(
  log_price_obs ~ dist_log +
    length + age_years + is_com + has_hyphen + has_digit +
    token_count + exact_match |
    cluster_keyword,
  data = cross_sold, vcov = "hetero"
)

save_models(
  list("Linear" = m2_ff_lin, "Quadratic" = m2_ff_quad, "Log(1+d)" = m2_ff_log),
  "tab_part2_functional_forms",
  title = "Appendix — Part II Robustness: Functional Form"
)

# Heterogeneity splits: TLD, age, length
# Split by .com vs non-.com
m2_com <- feols(
  log_price_obs ~ distance_used +
    length + age_years + has_hyphen + has_digit + token_count + exact_match |
    cluster_keyword,
  data = filter(cross_sold, is_com == 1),
  vcov = "hetero"
)

m2_noncom <- feols(
  log_price_obs ~ distance_used +
    length + age_years + has_hyphen + has_digit + token_count + exact_match |
    cluster_keyword,
  data = filter(cross_sold, is_com == 0),
  vcov = "hetero"
)

# Age split at median
age_med <- median(cross_sold$age_years, na.rm = TRUE)

m2_young <- feols(
  log_price_obs ~ distance_used +
    length + age_years + is_com + has_hyphen + has_digit + token_count + exact_match |
    cluster_keyword,
  data = filter(cross_sold, age_years <= age_med),
  vcov = "hetero"
)

m2_old <- feols(
  log_price_obs ~ distance_used +
    length + age_years + is_com + has_hyphen + has_digit + token_count + exact_match |
    cluster_keyword,
  data = filter(cross_sold, age_years > age_med),
  vcov = "hetero"
)

# Length split at median
len_med <- median(cross_sold$length, na.rm = TRUE)

m2_short <- feols(
  log_price_obs ~ distance_used +
    length + age_years + is_com + has_hyphen + has_digit + token_count + exact_match |
    cluster_keyword,
  data = filter(cross_sold, length <= len_med),
  vcov = "hetero"
)

m2_long <- feols(
  log_price_obs ~ distance_used +
    length + age_years + is_com + has_hyphen + has_digit + token_count + exact_match |
    cluster_keyword,
  data = filter(cross_sold, length > len_med),
  vcov = "hetero"
)

save_models(
  list(
    ".com only" = m2_com,
    "Non-.com" = m2_noncom,
    "Younger domains" = m2_young,
    "Older domains" = m2_old,
    "Shorter domains" = m2_short,
    "Longer domains" = m2_long
  ),
  "tab_part2_heterogeneity",
  title = "Appendix — Part II Robustness: Heterogeneity by TLD, Age, and Length"
)

# Additional descriptive tables
desc_part1 <- chat_sales %>%
  summarise(
    n_obs = n(),
    n_domains = n_distinct(domain),
    mean_log_price = mean(log_price_obs, na.rm = TRUE),
    sd_log_price = sd(log_price_obs, na.rm = TRUE),
    mean_distance = mean(distance_used, na.rm = TRUE)
  )

write_csv(desc_part1, file.path(out_tab, "desc_part1_chat_sales.csv"))

desc_part2 <- cross_sold %>%
  group_by(cluster_keyword) %>%
  summarise(
    n_obs = n(),
    mean_log_price = mean(log_price_obs, na.rm = TRUE),
    sd_log_price = sd(log_price_obs, na.rm = TRUE),
    mean_distance = mean(distance_used, na.rm = TRUE),
    mean_Sk = mean(S_k, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(desc_part2, file.path(out_tab, "desc_part2_by_cluster.csv"))

# Summary
cat("\nSaved tables to:", out_tab, "\n")
cat("Saved figures to:", out_fig, "\n")

cat("\nPart I main FE model:\n")
print(summary(m1_fe))

cat("\nPart I event-study model:\n")
print(etable(m_es))

cat("\nPart II pooled model:\n")
print(summary(m2_pooled))

cat("\nPart II popularity-gradient second stage:\n")
print(summary(m2_pop))














