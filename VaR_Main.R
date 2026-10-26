library(quantmod)
library(zoo)
library(tseries)

### Step 0: Load and validate the fixed S&P 500 input

# The analysis uses the archived CSV as its authoritative input. It is not
# downloaded or overwritten during a reproducible run: a live Yahoo download
# could be revised later and would invalidate the separately trained LSTM
# forecasts. The study covers the 2004-2024 calendar period; the archived file's
# final available observation is 2024-12-30 and this is reported explicitly.

### CHECK DATASET AVAILABILITY BEFORE MOVING TO ANALYSIS CASES BELOW CHECK!

data_file <- "sp500_2004_2024.csv"
if (!file.exists(data_file)) {
  stop("Missing required input: ", data_file)
}

sp500_df <- read.csv(data_file, stringsAsFactors = FALSE)
required_price_cols <- c("Date", "GSPC.Close")
if (!all(required_price_cols %in% names(sp500_df))) {
  stop("S&P 500 input must contain: ", paste(required_price_cols, collapse = ", "))
}

sp500_df$Date <- as.Date(sp500_df$Date)
if (anyNA(sp500_df[, required_price_cols])) stop("S&P 500 input contains missing values.")
if (anyDuplicated(sp500_df$Date)) stop("S&P 500 input contains duplicate dates.")
if (is.unsorted(sp500_df$Date, strictly = TRUE)) stop("S&P 500 dates are not strictly increasing.")

cat("S&P 500 input:", nrow(sp500_df), "prices from",
    format(min(sp500_df$Date)), "to", format(max(sp500_df$Date)), "\n")

### REPRODUCIBILITY SEED SET.

set.seed(123)

library(quantmod)
library(zoo)
library(tseries)

### Step 1: Cleaning and Return Construction: We will Discuss the stationarity of the time-series data. 
# then Transform the data into logreturns. Make time plots.

# No Transformations yet
plot(sp500_df$Date, sp500_df$GSPC.Close, type = "l", 
     main = "S&P 500 Index Closing Prices",
     xlab = "Date", ylab = "Price")

# Compute log returns
sp500_df$Log_Returns <- c(NA, diff(log(sp500_df$GSPC.Close)))

# Remove NA values (first row)
sp500_df <- na.omit(sp500_df) # Cleaning

# Plot Log Returns
plot(sp500_df$Date, sp500_df$Log_Returns, type = "l",
     main = "S&P 500 Index Log Returns",
     xlab = "Date", ylab = "Log Returns")

adf.test(sp500_df$Log_Returns) # A small p-value rejects the unit-root null.

### Step 2: Study the distribution of returns. 
### (volatility clustering, distributional properties, kurtosis, skewness)

library(TSA)

# 1st to 4th Moments
# Compute mean, variance, skewness, and kurtosis
mean_return <- mean(sp500_df$Log_Returns)
var_return <- var(sp500_df$Log_Returns)
skewness_return <- skewness(sp500_df$Log_Returns)
kurtosis_return <- kurtosis(sp500_df$Log_Returns)

# Mean is very low very close to 0.
# skewness is quite weak at approx. -0.53 which is below the 
# 1 and -1 threshold. But We have a kurtosis of above 3 at 
# approx. 13.3 showing Heavy Tails.

hist(sp500_df$Log_Returns, breaks = 50, probability = TRUE,
     main = "Histogram of S&P 500 Log Returns", xlab = "Log Returns")

# Overlay Normal Distribution
curve(dnorm(x, mean = mean_return, sd = sqrt(var_return)), col = "red", lwd = 2, add = TRUE)

# Generate a QQ Plot
qqnorm(sp500_df$Log_Returns, main = "QQ Plot of S&P 500 Log Returns",
       xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
qqline(sp500_df$Log_Returns, col = "red", lwd = 2)  # Add reference line

# Heavier Tails can be seen from the QQ-plot.

# Test normality using the Jarque-Bera test. A two-sample K-S test against a
# simulated normal series is intentionally not used: its result depends on the
# simulated comparison sample and is not the appropriate fitted-normal test.
jarque.bera.test(sp500_df$Log_Returns)
# At the 5% level, the small p-value rejects normality. This formal result is
# interpreted together with the QQ plot, skewness and kurtosis.

# We can see faintly against the normal distribution the Heavy Tail
# Phenomena for Returns.

# Volatility Clustering

# Compute Squared Log Returns
sp500_df$Squared_Returns <- sp500_df$Log_Returns^2

# Plot Squared Log Returns
plot(sp500_df$Date, sp500_df$Squared_Returns, type = "l", col = "black",
     xlab = "Date", ylab = "Squared Log Returns",
     main = "Squared Log Returns of S&P 500 Index")

# From this plot we can now see Volatility Clustering quite clearly.
# the squared returns plot show periods of high variance followed by low variance.

# Fit GARCH(1,1) model to S&P500 log returns using tseries
garch_model <- garch(sp500_df$Log_Returns, order = c(1,1))

summary(garch_model)

# Coefficient estimates and p-values are inspected as descriptive evidence;
# this full-sample GARCH fit is not one of the rolling VaR forecasts.

# Residuals Plot
plot(resid(garch_model), type = "l", col = "black",
     main = "GARCH(1,1) Model Residuals",
     ylab = "Residuals", xlab = "Time")

# QQ Plot of Residuals
qqnorm(resid(garch_model))
qqline(resid(garch_model), col = "red")  # Add reference line

# Conditional Variance Plot
plot(fitted(garch_model)[,1]^2, type = "l", col = "blue",
     main = "Conditional Variance from GARCH(1,1)",
     ylab = "Estimated Conditional Variance", xlab = "Time")

# This plot shows how risk is changing over time.

# There are spikes in conditional variance, indicating volatility 
# clustering—periods of high volatility are followed by more high volatility, 
# and periods of low volatility are followed by low volatility.
# The GARCH(1,1) model successfully captures changing variance, confirming 
# that log returns exhibit conditional heteroskedasticity, meaning the variance is 
# time-dependent.

acf(sp500_df$Log_Returns, main="ACF of Returns")
acf(sp500_df$Log_Returns^2, main="ACF of Squared Returns")

# the ACF of squared returns displays strong and persistent positive autocorrelation, 
# providing clear evidence of volatility clustering.

### Step 3: Calculate Value at Risk at different confidence (Baseline VaR models)

## Baseline - Using the Whole Dataset. (Sanity Check)

library(PerformanceAnalytics)

# Since the data is not normal lets use modified and historical
# VaR computations.

# Define confidence levels
confidence_levels <- c(0.90, 0.95, 0.99)

# Compute Gaussian Parametric VaR
VaR_gaussian <- sapply(confidence_levels, function(p) {
  VaR(sp500_df$Log_Returns, p = p, method = "gaussian")
})

# Compute Historical VaR
VaR_historical <- sapply(confidence_levels, function(p) {
  VaR(sp500_df$Log_Returns, p = p, method = "historical")
})

# Compute Modified VaR (Cornish-Fisher)
VaR_modified <- sapply(confidence_levels, function(p) {
  VaR(sp500_df$Log_Returns, p = p, method = "modified")
})

# Format results into a data frame
VaR_results <- data.frame(
  Confidence_Level = confidence_levels,
  Parametric_VaR = VaR_gaussian,
  Historical_VaR = VaR_historical,
  Modified_VaR = VaR_modified
)

# Print the formatted table
print(VaR_results)

# The Value at Risk (VaR) results for the S&P500's log returns for 2004 to 2024
# indicate the potential worst-case losses at different confidence levels. 
# These in-sample values are descriptive sanity checks, not forecasts or
# backtests. At 99%, Historical VaR is about -3.50% and Modified VaR about
# -6.78%; the larger Modified loss threshold reflects the sample's skewness and
# kurtosis. Whether that additional conservatism improves forecast calibration
# is assessed only by the rolling out-of-sample backtests below.

### Step 4: Crisis Window Selection and Justifying the COVID Shock Crisis Window.

# Non-Crisis Period (2008 Crisis) - January 1, 2004 – December 31, 2006
# Crisis Period (2008 Crisis) - August 1, 2007 – June 30, 2009
# Non-Crisis Period (COVID) - January 1, 2017 – December 31, 2019
# Crisis Period (COVID) - February 1, 2020 – April 30, 2020. This deliberately
# short shock window captures the abrupt volatility spike but gives low-powered
# formal backtests, which is disclosed alongside the results.

library(zoo)

# Use a short 20-trading-day window to show the local evolution of volatility.

# 20-day rolling volatility
sp500_df$RollingVol20 <- rollapply(sp500_df$Log_Returns, width = 20, FUN = sd, align = "right", fill = NA)

# Plot rolling volatility
plot(sp500_df$Date, sp500_df$RollingVol20, type = "l",
     main = "20-Day Rolling Volatility of S&P 500 Returns",
     xlab = "Date", ylab = "Rolling Volatility")

# Mark COVID window
abline(v = as.Date("2020-02-01"), col = "red", lwd = 2)
abline(v = as.Date("2020-04-30"), col = "red", lwd = 2)

# Zoom in a little more

covid_start <- as.Date("2019-09-01")
covid_end <- as.Date("2020-06-30")

subset <- sp500_df[sp500_df$Date >= covid_start & sp500_df$Date <= covid_end, ]

plot(subset$Date, subset$RollingVol20, type = "l", lwd = 2,
     main = "20-Day Rolling Volatility Around the COVID Shock",
     xlab = "Date", ylab = "Rolling Volatility")

abline(v = as.Date("2007-08-01"), col = "red", lwd = 2)
abline(v = as.Date("2009-06-30"), col = "red", lwd = 2)

abline(v = as.Date("2020-02-01"), col = "red", lwd = 2)
abline(v = as.Date("2020-04-30"), col = "red", lwd = 2)

plot(sp500_df$Date, sp500_df$RollingVol20, type = "l",
     main = "20-Day Rolling Volatility of S&P 500 Returns",
     xlab = "Date", ylab = "Rolling Volatility")

# A 20-day rolling volatility measure shows a sharp increase in volatility beginning 
# in late February 2020 and peaking in mid-March. The crisis window of February 1 to April 30 
# captures the period of extreme volatility and the subsequent stabilization phase.

# It would be better to use Feb 20 to June 30. Longer Window but captures stabilization better.

# Test in the 2008 Crisis Just to see difference.
crisis2008_start <- as.Date("2006-01-01")
crisis2008_end   <- as.Date("2010-12-31")

subset2008 <- sp500_df[sp500_df$Date >= crisis2008_start & sp500_df$Date <= crisis2008_end, ]

plot(subset2008$Date, subset2008$RollingVol20, type = "l", lwd = 2,
     main = "20-Day Rolling Volatility Around the 2008 Financial Crisis",
     xlab = "Date", ylab = "Rolling Volatility")

abline(v = as.Date("2007-08-01"), col = "red", lwd = 2)
abline(v = as.Date("2009-06-30"), col = "red", lwd = 2)

### Step 5: Baseline Models with Roll forward forecasting mechanism.
## Note that we will use 99 Confidence level on all models to ensure comparability.

window_size <- 127 # Primary common estimation window used by all rolling R models.
n <- nrow(sp500_df)

VaR_Hist_99  <- rep(NA, n)
VaR_Gauss_99 <- rep(NA, n)
VaR_Mod_99   <- rep(NA, n)

for (t in (window_size + 1):n) {
  
  window_data <- sp500_df$Log_Returns[(t - window_size):(t - 1)]
  
  VaR_Hist_99[t]  <- VaR(window_data, p = 0.99, method = "historical")
  VaR_Gauss_99[t] <- VaR(window_data, p = 0.99, method = "gaussian")
  VaR_Mod_99[t]   <- VaR(window_data, p = 0.99, method = "modified")
}

results_df <- data.frame(
  Row_ID = seq_len(n),
  Date = sp500_df$Date,
  Return = sp500_df$Log_Returns,
  VaR_Hist_99 = VaR_Hist_99,
  VaR_Gauss_99 = VaR_Gauss_99,
  VaR_Mod_99 = VaR_Mod_99
)

results_df$Viol_Hist_99  <- ifelse(results_df$Return < results_df$VaR_Hist_99, 1, 0)
results_df$Viol_Gauss_99 <- ifelse(results_df$Return < results_df$VaR_Gauss_99, 1, 0)
results_df$Viol_Mod_99   <- ifelse(results_df$Return < results_df$VaR_Mod_99, 1, 0)

# Sanity check for violation results.
mean(results_df$Viol_Hist_99, na.rm = TRUE)
mean(results_df$Viol_Gauss_99, na.rm = TRUE)
mean(results_df$Viol_Mod_99, na.rm = TRUE)

## Rates above the 1% target indicate an anti-conservative VaR boundary. Formal
## significance is assessed below rather than inferred from rates alone.

## We will label the Time Stamps as Calm or GFC or COVID

results_df$Regime <- "Calm"

results_df$Regime[
  results_df$Date >= "2007-08-01" & results_df$Date <= "2009-06-30"
] <- "GFC"

results_df$Regime[
  results_df$Date >= "2020-02-01" & results_df$Date <= "2020-04-30"
] <- "COVID"

# Sanity check
head(results_df, 140)
table(results_df$Regime)

colnames(results_df)

## Kupiec Test

kupiec_test <- function(violations, p = 0.01) {
  
  violations <- na.omit(violations)
  
  T <- length(violations)
  N <- sum(violations)
  
  p_hat <- N / T
  
  # x*log(y) is defined as zero when x=0. Handling those boundary cases
  # explicitly keeps the test valid when a short window has zero violations.
  xlog <- function(x, y) ifelse(x == 0, 0, x * log(y))
  logL_null <- xlog(T - N, 1 - p) + xlog(N, p)
  logL_alt  <- xlog(T - N, 1 - p_hat) + xlog(N, p_hat)
  LR <- -2 * (logL_null - logL_alt)
  
  p_value <- pchisq(LR, df = 1, lower.tail = FALSE)
  
  # FOR CONVENIENCE - OUTPUTS ALL WE NEED
  return(list(
    Observations = T,
    Violations = N,
    Violation_Rate = p_hat,
    LR_stat = LR,
    p_value = p_value
  ))
}

# Compute all three standard coverage diagnostics. Christoffersen conditional
# coverage is the joint test LR_CC = LR_UC + LR_IND. LR_IND is also reported
# separately so a joint rejection is not automatically described as evidence
# of clustering. Row_ID prevents transitions from being counted across dates
# removed because a model forecast was unavailable.

### NOTE to self: New style -> Lets use NaN declaration with type hinting to improve run stability.

### IMPORTANT NOTE: 

# Issues encountered with gaps and sometimes the next date after blanks is treated as consecutive.

### SOLUTION THEREFORE:

# Function below implements the standard Kupiec unconditional-coverage,
# Christoffersen independence, and joint conditional-coverage tests directly
# as opposed to packages; A custom implementation is used because the common-date samples can 
# contain gaps  when a model forecast is unavailable. Many package functions treat adjacent rows 
# in the filtered data as consecutive observations, even when trading dates were omitted. 
# By using Row_ID, the independence test counts transitions only between observations
# that were consecutive in the original return series. The function also
# provides consistent violation counts, expected counts, p-values, decisions,
# and transition diagnostics for every model and evaluation window.

# Variable style uses Christoffersen (1998) paper. And some style inspiration from
# https://gist.github.com/agabrielsen and https://github.com/YujianCHEN219/ExactVaRTest
# for code interpret guidance to decrypt the equation. Quite tricky.

coverage_tests <- function(data, var_col, alpha = 0.01) {
  d <- data[!is.na(data$Return) & !is.na(data[[var_col]]), ]
  violations <- as.integer(d$Return < d[[var_col]])
  T_obs <- length(violations)

  if (T_obs == 0) {
    return(list(
      N = 0, Violations = NA_integer_, Expected = 0, Violation_Rate = NA_real_,
      LR_UC = NA_real_, p_UC = NA_real_, Decision_UC = NA_character_,
      LR_IND = NA_real_, p_IND = NA_real_, Decision_IND = NA_character_,
      LR_CC = NA_real_, p_CC = NA_real_, Decision_CC = NA_character_,
      Valid_Transitions = 0L, Contiguous_Runs = 0L
    ))
  }

  xlog <- function(x, y) ifelse(x == 0, 0, x * log(y))
  n_viol <- sum(violations)
  p_hat <- n_viol / T_obs
  logL_uc_null <- xlog(T_obs - n_viol, 1 - alpha) + xlog(n_viol, alpha)
  logL_uc_alt <- xlog(T_obs - n_viol, 1 - p_hat) + xlog(n_viol, p_hat)
  LR_UC <- max(0, -2 * (logL_uc_null - logL_uc_alt))
  p_UC <- pchisq(LR_UC, df = 1, lower.tail = FALSE)

  valid_pair <- if (T_obs >= 2) diff(d$Row_ID) == 1 else logical(0)
  previous <- if (T_obs >= 2) violations[-T_obs][valid_pair] else integer(0)
  current  <- if (T_obs >= 2) violations[-1][valid_pair] else integer(0)

  n00 <- sum(previous == 0 & current == 0)
  n01 <- sum(previous == 0 & current == 1)
  n10 <- sum(previous == 1 & current == 0)
  n11 <- sum(previous == 1 & current == 1)
  n_trans <- n00 + n01 + n10 + n11

  LR_IND <- p_IND <- NA_real_
  if (n_trans > 0) {
    pi_all <- (n01 + n11) / n_trans
    pi_01 <- if ((n00 + n01) > 0) n01 / (n00 + n01) else NA_real_
    pi_11 <- if ((n10 + n11) > 0) n11 / (n10 + n11) else NA_real_

    if (!is.na(pi_01) && !is.na(pi_11)) {
      logL_ind_null <- xlog(n00 + n10, 1 - pi_all) + xlog(n01 + n11, pi_all)
      logL_ind_alt <- xlog(n00, 1 - pi_01) + xlog(n01, pi_01) +
        xlog(n10, 1 - pi_11) + xlog(n11, pi_11)
      LR_IND <- max(0, -2 * (logL_ind_null - logL_ind_alt))
      p_IND <- pchisq(LR_IND, df = 1, lower.tail = FALSE)
    }
  }

  LR_CC <- if (is.na(LR_IND)) NA_real_ else LR_UC + LR_IND
  p_CC <- if (is.na(LR_CC)) NA_real_ else pchisq(LR_CC, df = 2, lower.tail = FALSE)
  decision <- function(p_value) {
    if (is.na(p_value)) NA_character_ else if (p_value < 0.05) "Reject H0" else "Fail to Reject H0"
  }

  list(
    N = T_obs,
    Violations = n_viol,
    Expected = alpha * T_obs,
    Violation_Rate = p_hat,
    LR_UC = LR_UC, p_UC = p_UC, Decision_UC = decision(p_UC),
    LR_IND = LR_IND, p_IND = p_IND, Decision_IND = decision(p_IND),
    LR_CC = LR_CC, p_CC = p_CC, Decision_CC = decision(p_CC),
    Valid_Transitions = n_trans,
    Contiguous_Runs = if (T_obs == 0) 0L else 1L + sum(diff(d$Row_ID) != 1)
  )
}

backtest_row <- function(data, model, var_col, alpha = 0.01) {
  bt <- coverage_tests(data, var_col, alpha)
  data.frame(
    Model = model,
    N = bt$N,
    Violations = bt$Violations,
    Expected = bt$Expected,
    Violation_Rate = bt$Violation_Rate,
    Kupiec_LR = bt$LR_UC,
    Kupiec_p = bt$p_UC,
    Kupiec = bt$Decision_UC,
    Independence_LR = bt$LR_IND,
    Independence_p = bt$p_IND,
    Independence = bt$Decision_IND,
    Conditional_Coverage_LR = bt$LR_CC,
    Conditional_Coverage_p = bt$p_CC,
    Conditional_Coverage = bt$Decision_CC,
    Valid_Transitions = bt$Valid_Transitions,
    Contiguous_Runs = bt$Contiguous_Runs,
    stringsAsFactors = FALSE
  )
}

colMeans(results_df[, c(
  "Viol_Hist_99",
  "Viol_Gauss_99",
  "Viol_Mod_99"
)], na.rm = TRUE)

# Whole Sample

kupiec_test(results_df$Viol_Hist_99)
kupiec_test(results_df$Viol_Gauss_99)
kupiec_test(results_df$Viol_Mod_99)

# Per Crisis Regime

kupiec_test(subset(results_df, Regime == "GFC")$Viol_Hist_99)
kupiec_test(subset(results_df, Regime == "GFC")$Viol_Gauss_99)
kupiec_test(subset(results_df, Regime == "GFC")$Viol_Mod_99)

kupiec_test(subset(results_df, Regime == "COVID")$Viol_Hist_99)
kupiec_test(subset(results_df, Regime == "COVID")$Viol_Gauss_99)
kupiec_test(subset(results_df, Regime == "COVID")$Viol_Mod_99)

# Inspect the reported p-values before drawing regime-specific conclusions.

aggregate(. ~ Regime, data = results_df[, c(
  "Regime",
  "Viol_Hist_99",
  "Viol_Gauss_99",
  "Viol_Mod_99"
)], mean, na.rm = TRUE)


# These preliminary summaries use each baseline model's available observations.
# The exported tables provide exact counts and formal decisions, while the
# common-sample analysis below supplies the primary cross-model comparison.

## Coverage diagnostics from rugarch

# Remove NAs of first 127 rows for rolling VaR.
df_hist <- na.omit(results_df[, c("Return", "VaR_Hist_99")])
df_gauss <- na.omit(results_df[, c("Return", "VaR_Gauss_99")])
df_mod   <- na.omit(results_df[, c("Return", "VaR_Mod_99")])

library(rugarch)

VaRTest(alpha = 0.01,
        actual = df_hist$Return,
        VaR = df_hist$VaR_Hist_99)

# `cc` is the joint conditional-coverage result. It does not, on its own,
# identify whether rejection is caused by frequency, dependence, or both.

VaRTest(alpha = 0.01,
        actual = df_gauss$Return,
        VaR = df_gauss$VaR_Gauss_99)

# Interpret the unconditional and independence components separately below.

VaRTest(alpha = 0.01,
        actual = df_mod$Return,
        VaR = df_mod$VaR_Mod_99)

# The formal tables below report Kupiec unconditional coverage,
# Christoffersen independence, and their joint conditional-coverage test.

### Step 6: Econometrics Models with Roll forward forecasting mechanism.
## Models that capture time-varying volatility, asymmetry and tail behaviour.

library(rugarch)


## EGARCH Model

# Fit one illustrative full-sample model as a specification check.
spec <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(0,0)),
  distribution.model = "std"   # Student-t (important)
)

fit <- ugarchfit(spec, data = sp500_df$Log_Returns)

forecast <- ugarchforecast(fit, n.ahead = 1)

mu <- fitted(forecast)
sigma <- sigma(forecast)

VaR_egarch <- mu + sigma * qdist("std", p = 0.01,
                                 mu = 0,
                                 sigma = 1,
                                 shape = coef(fit)["shape"])

VaR_egarch # Specification check only; the dated backtest uses the rolling loop.

## Full Loop Simulation

window_size <- 127
n <- nrow(results_df)

VaR_EGARCH_99 <- rep(NA_real_, n) ## FORCE TO NUMERICAL

spec <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "std"
)

# Re-seed immediately before the loop that can trigger gosolnp's random
# restarts, so the loop's results are fixed regardless of what runs earlier.
set.seed(123)

for (t in (window_size + 1):n) {
  
  # use only past data
  window_data <- results_df$Return[(t - window_size):(t - 1)]
  window_data <- na.omit(window_data)
  
  # skip if not enough usable observations
  if (length(window_data) < 100) {
    next
  }
  
  # fit EGARCH safely
  # NOTE ON REPRODUCIBILITY: solver = "hybrid" falls back to Rsolnp::gosolnp
  # when "solnp" fails to converge. gosolnp's own source sets
  # `rseed <- as.numeric(Sys.time())` whenever rseed is NULL, which reseeds
  # the RNG from the system clock and bypasses any set.seed() in this script
  # entirely. Passing rseed explicitly here (rather than relying on the
  # ambient RNG state) is what actually makes every re-run deterministic.
  fit <- tryCatch(
    ugarchfit(
      spec = spec,
      data = window_data,
      solver = "hybrid",
      solver.control = list(trace = 0, rseed = 123)
    ),
    error = function(e) NULL
  )
  
  # skip if fit failed
  if (is.null(fit)) {
    next
  }
  
  # 1-step-ahead forecast safely
  fcst <- tryCatch(
    ugarchforecast(fit, n.ahead = 1),
    error = function(e) NULL
  )
  
  # skip if forecast failed
  if (is.null(fcst)) {
    next
  }
  
  mu_t <- as.numeric(fitted(fcst))
  sigma_t <- as.numeric(sigma(fcst))
  shape_t <- as.numeric(coef(fit)["shape"])
  
  # skip if anything is missing or invalid
  if (any(is.na(c(mu_t, sigma_t, shape_t))) || sigma_t <= 0 || shape_t <= 2) {
    next
  }
  
  VaR_EGARCH_99[t] <- mu_t + sigma_t * qdist(
    "std",
    p = 0.01,
    mu = 0,
    sigma = 1,
    shape = shape_t
  )
}

# add to master table
results_df$VaR_EGARCH_99 <- VaR_EGARCH_99

# violations
results_df$Viol_EGARCH_99 <- ifelse(
  results_df$Return < results_df$VaR_EGARCH_99,
  1,
  0
)

# Sanity Check
sum(!is.na(results_df$VaR_EGARCH_99))
head(results_df[, c("Date", "Return", "VaR_EGARCH_99", "Viol_EGARCH_99")], 140)
mean(results_df$Viol_EGARCH_99, na.rm = TRUE)

# Forecast count and violation rate are printed as implementation checks. Their
# statistical interpretation is deferred to the decomposed tests below.

df_egarch <- na.omit(results_df[, c("Return", "VaR_EGARCH_99")])

VaRTest(
  alpha = 0.01,
  actual = df_egarch$Return,
  VaR = df_egarch$VaR_EGARCH_99
)

# The fixed 127-day window is intentionally shared with the baseline and EVT
# models for comparability, but it is short for EGARCH estimation. Results
# should therefore be interpreted as performance under this common constraint,
# not as the best performance EGARCH could attain with a longer history.

# The EGARCH violation rate still exceeds 1%. The full diagnostics below
# determine separately whether its failures concern unconditional coverage,
# independence, or both; a joint conditional-coverage rejection alone is not
# described as proof of clustering.

# EVT/POT is added next to test whether explicit tail modelling improves the
# one-day 99% boundary under the same rolling-window constraint.

## EVT VaR Model

library(evir)

# Fit one illustrative rolling window as a specification check.

window_size <- 127
t <- 300

# Use past data only
window_data <- results_df$Return[(t - window_size):(t - 1)]
window_data <- na.omit(window_data)

length(window_data)
summary(window_data)

# convert returns to positive losses
losses <- -window_data

threshold <- quantile(losses, 0.90, na.rm = TRUE) # About 13 exceedances in 127 observations.

threshold
sum(losses > threshold)

evt_fit <- gpd(losses, threshold) # Generalized Pareto Distribution is used.

summary(evt_fit)

# Then compute 99% EVT VaR

p <- 0.99

n_loss <- length(losses)
Nu <- sum(losses > threshold)

xi <- evt_fit$par.ests["xi"]
beta <- evt_fit$par.ests["beta"]

VaR_loss <- threshold + (beta / xi) * (
  (((n_loss / Nu) * (1 - p))^(-xi)) - 1
)

VaR_EVT_99 <- -VaR_loss

VaR_EVT_99

# compare it with the actual return at t

results_df[t, c("Date", "Return")]

ifelse(results_df$Return[t] < VaR_EVT_99, 1, 0) # One-window implementation check.

# Due to the relatively short rolling estimation window of 127 observations, a 90th percentile 
# threshold was selected for the Peaks Over Threshold (POT) method to ensure a sufficient number 
# of exceedances for stable Generalized Pareto Distribution estimation.

### WE WILL RUN THROUGH ALL THE MODELS APPLICABLE IN R HERE USING LOOP SIM BELOW.

## Full Loop Simulation

# 90% POT threshold for GPD and 99% VaR.

window_size <- 127
n <- nrow(results_df)

VaR_EVT_99 <- rep(NA_real_, n)
evt_fit_error_count <- 0L
evt_warning_count <- 0L
evt_optimization_warning_count <- 0L

for (t in (window_size + 1):n) {
  
  # Use only past returns
  window_data <- results_df$Return[(t - window_size):(t - 1)]
  window_data <- na.omit(window_data)
  
  # Skip short windows
  if (length(window_data) < 100) {
    next
  }
  
  # Convert returns to positive losses
  # Keep the FULL loss distribution (both up- and down-days). Filtering to
  # losses > 0 would truncate the sample, so n_loss and the 90th-percentile
  # threshold would be based only on down-days, distorting the n/Nu ratio in
  # the PoT VaR formula. This matches the t = 300 sanity check above.
  losses <- -window_data
  losses <- losses[is.finite(losses)]
  
  if (length(losses) < 50) {
    next
  }
  
  # POT threshold: top 10% of losses
  threshold <- as.numeric(quantile(losses, 0.90, na.rm = TRUE))
  
  # Need enough exceedances
  Nu <- sum(losses > threshold)
  
  if (Nu < 5) {
    next
  }
  
  # Fit GPD safely
  fit_warnings <- character(0)
  evt_fit <- tryCatch(
    withCallingHandlers(
      gpd(losses, threshold),
      warning = function(w) {
        fit_warnings <<- c(fit_warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      evt_fit_error_count <<- evt_fit_error_count + 1L
      NULL
    }
  )
  evt_warning_count <- evt_warning_count + length(fit_warnings)
  evt_optimization_warning_count <- evt_optimization_warning_count +
    sum(grepl("optimization may not have succeeded", fit_warnings, fixed = TRUE))
  
  if (is.null(evt_fit)) {
    next
  }
  
  # Extract GPD parameters
  xi <- as.numeric(evt_fit$par.ests["xi"])
  beta <- as.numeric(evt_fit$par.ests["beta"])
  
  if (any(is.na(c(xi, beta))) || beta <= 0) {
    next
  }
  
  # Guard against implausible / unstable tail-shape estimates from the small
  # 127-day window. xi >= 1 implies an infinite-mean GPD where the VaR is
  # explosive/unreliable, and xi < -0.5 violates the regularity conditions
  # under which GPD MLE is well-behaved (Smith, 1985). Such windows are
  # skipped rather than trusted.
  if (xi >= 1 || xi < -0.5) {
    next
  }
  
  # Compute 99% EVT VaR in loss space
  p <- 0.99
  n_loss <- length(losses)
  
  if (abs(xi) < 1e-6) {
    # Approximation when xi is close to zero
    VaR_loss <- threshold + beta * log((n_loss / Nu) * (1 / (1 - p)))
  } else {
    VaR_loss <- threshold + (beta / xi) * (
      (((n_loss / Nu) * (1 - p))^(-xi)) - 1
    )
  }
  
  # Final sanity check: a loss-space VaR must be finite and positive
  # (threshold plus a non-negative tail term). Anything else signals a
  # degenerate fit and is discarded to avoid being blindsided by outliers.
  if (!is.finite(VaR_loss) || VaR_loss <= 0) {
    next
  }
  
  # Convert back to return space
  VaR_EVT_99[t] <- -VaR_loss
}

# Add to results table
results_df$VaR_EVT_99 <- VaR_EVT_99

# Add violation indicator
results_df$Viol_EVT_99 <- ifelse(
  results_df$Return < results_df$VaR_EVT_99,
  1,
  0
)

# Sanity Checks
sum(!is.na(results_df$VaR_EVT_99))
mean(results_df$Viol_EVT_99, na.rm = TRUE)
cat("EVT fit errors:", evt_fit_error_count,
    "| warnings captured:", evt_warning_count,
    "| optimization warnings:", evt_optimization_warning_count, "\n")

# A missing forecast is retained as missing rather than replaced with an
# unreliable tail estimate. Forecast availability is reported explicitly: the
# high failure rate is an operational limitation of applying rolling POT/GPD to
# only 127 observations and can also create non-contiguous backtest segments.

head(results_df[, c(
  "Date", "Return", "VaR_EVT_99", "Viol_EVT_99"
)], 140)

df_evt <- na.omit(results_df[, c("Return", "VaR_EVT_99")])

VaRTest(
  alpha = 0.01,
  actual = df_evt$Return,
  VaR = df_evt$VaR_EVT_99
)

# EVT fails both Kupiec unconditional coverage and Christoffersen joint
# conditional coverage on its available full-sample dates. Its ~2.25% rate is more
# than double the 1% target, so tail-focused POT modeling does not, on its
# own, fix full-sample coverage here. See Step 8 for the fair common-sample
# comparison across all 6 models, and Step 10 for how EVT performs
# specifically in calm vs. crisis windows.

# The LSTM provides the data-driven comparison and is estimated separately in
# the GPU-enabled notebook before its dated forecasts are imported below.

### Step 7: Long Short Term Memory (LSTM) VaR

nn_data <- results_df[, c("Date", "Return", "Regime")]
write.csv(nn_data, "sp500_lstm_input.csv", row.names = FALSE)

# IMPORT FROM PYTHON GPU ENV (COLAB)

# Read the walk-forward LSTM predictions exported by the Colab notebook.
lstm_preds <- read.csv("lstm_var_predictions_20_scaled.csv")

# Validate provenance before merging. This prevents predictions trained on a
# different source-data version from silently entering the R backtests.
required_lstm_cols <- c("Date", "Return", "VaR_LSTM_99", "Lookback", "Viol_LSTM_99")
if (!all(required_lstm_cols %in% names(lstm_preds))) {
  stop("LSTM predictions must contain: ", paste(required_lstm_cols, collapse = ", "))
}

lstm_preds$Date <- as.Date(lstm_preds$Date)
results_df$Date <- as.Date(results_df$Date)
if (anyNA(lstm_preds[, required_lstm_cols])) stop("LSTM prediction file contains missing values.")
if (anyDuplicated(lstm_preds$Date)) stop("LSTM prediction file contains duplicate dates.")
if (is.unsorted(lstm_preds$Date, strictly = TRUE)) stop("LSTM prediction dates are not strictly increasing.")

matched_rows <- match(lstm_preds$Date, results_df$Date)
if (anyNA(matched_rows)) stop("LSTM prediction file contains dates absent from the R return series.")
return_difference <- abs(lstm_preds$Return - results_df$Return[matched_rows])
if (any(return_difference > 1e-12)) {
  stop("LSTM prediction returns do not match the current R return series.")
}
recomputed_lstm_viol <- as.integer(lstm_preds$Return < lstm_preds$VaR_LSTM_99)
if (!identical(recomputed_lstm_viol, as.integer(lstm_preds$Viol_LSTM_99))) {
  stop("LSTM violation indicators are inconsistent with Return and VaR_LSTM_99.")
}

# Keep only needed columns
lstm_preds <- lstm_preds[, c("Date", "VaR_LSTM_99")]

# Remove old LSTM columns if they already exist
results_df$VaR_LSTM_99 <- NULL
results_df$Viol_LSTM_99 <- NULL

# Merge corrected predictions
results_df <- merge(results_df, lstm_preds, by = "Date", all.x = TRUE)

# Add violations
results_df$Viol_LSTM_99 <- ifelse(
  results_df$Return < results_df$VaR_LSTM_99,
  1,
  0
)

# Sanity checks
sum(!is.na(results_df$VaR_LSTM_99))
mean(results_df$Viol_LSTM_99, na.rm = TRUE)

head(results_df[!is.na(results_df$VaR_LSTM_99),
                c("Date", "Return", "VaR_LSTM_99", "Viol_LSTM_99")], 10)

df_lstm <- na.omit(results_df[, c("Return", "VaR_LSTM_99")])

VaRTest(
  alpha = 0.01,
  actual = df_lstm$Return,
  VaR = df_lstm$VaR_LSTM_99
)

### SUMMARY

model_cols <- c(
  Historical = "VaR_Hist_99",
  Gaussian   = "VaR_Gauss_99",
  Modified   = "VaR_Mod_99",
  EGARCH     = "VaR_EGARCH_99",
  EVT        = "VaR_EVT_99",
  LSTM       = "VaR_LSTM_99"
)

# Own-sample results describe each model wherever it produced a forecast.
# They are useful diagnostics but are not the primary cross-model comparison,
# because the EVT and LSTM availability windows differ from the other models.

comparison_table <- do.call(rbind, lapply(names(model_cols), function(m) {
  backtest_row(results_df, m, model_cols[[m]])
}))
print(comparison_table, row.names = FALSE)

# Report forecast availability separately rather than silently treating failed
# model fits as if those dates never existed. Baseline, EGARCH and EVT models
# are eligible after the 127-day rolling window; LSTM eligibility begins on its
# first exported walk-forward date after the notebook's 1000-day burn-in.

forecast_availability <- do.call(rbind, lapply(names(model_cols), function(m) {
  vcol <- model_cols[[m]]
  eligible <- if (m == "LSTM") {
    results_df$Date >= min(lstm_preds$Date) & results_df$Date <= max(lstm_preds$Date)
  } else {
    results_df$Row_ID > window_size
  }
  opportunities <- sum(eligible)
  available <- sum(eligible & !is.na(results_df[[vcol]]))
  data.frame(
    Model = m,
    Eligible_Dates = opportunities,
    Available_Forecasts = available,
    Missing_Forecasts = opportunities - available,
    Availability_Rate = available / opportunities,
    stringsAsFactors = FALSE
  )
}))
print(forecast_availability, row.names = FALSE)

### Step 8: Common-Sample (Intersection-of-Dates) Backtest

# The comparison_table above scores each model on its OWN available dates
# (per-model na.omit). That is not directly comparable: the EVT model drops
# windows where stable GPD estimation fails, and the LSTM only begins after its
# burn-in period. To compare like-for-like we restrict every model to the
# IDENTICAL set of dates on which all relevant models produce a VaR, then re-run
# the same Kupiec (unconditional) and Christoffersen (conditional) coverage
# tests. We report two shared samples:
#   (A) the five model-based series excluding the LSTM (widest comparable set), and
#   (B) all six models including the LSTM (strict intersection, shorter sample).

# Helper: backtest each model over a shared data subset. The supplied rows are
# identical across models; Row_ID still ensures the independence statistic does
# not invent transitions across dates excluded from the intersection.

common_backtest <- function(data, models) {
  do.call(rbind, lapply(names(models), function(m) {
    backtest_row(data, m, models[[m]])
  }))
}

# (A) Common sample EXCLUDING the LSTM (widest comparable window)
cols_noLSTM   <- model_cols[c("Historical", "Gaussian", "Modified", "EGARCH", "EVT")]
mask_noLSTM   <- complete.cases(results_df[, c("Return", cols_noLSTM)])
common_noLSTM <- results_df[mask_noLSTM, ]

cat("\n=== Common sample (no LSTM) ===\n")
cat("Dates:", format(min(common_noLSTM$Date)), "->", format(max(common_noLSTM$Date)),
    "| n =", nrow(common_noLSTM), "\n")
backtest_noLSTM <- common_backtest(common_noLSTM, cols_noLSTM)
print(backtest_noLSTM, row.names = FALSE)

# (B) Common sample INCLUDING the LSTM (strict six-way intersection)
mask_all   <- complete.cases(results_df[, c("Return", model_cols)])
common_all <- results_df[mask_all, ]

cat("\n=== Common sample (all 6 incl. LSTM) ===\n")
cat("Dates:", format(min(common_all$Date)), "->", format(max(common_all$Date)),
    "| n =", nrow(common_all), "\n")
backtest_all <- common_backtest(common_all, model_cols)
print(backtest_all, row.names = FALSE)

### Step 9: Results Visualizations

# The VaR literature's standard results-section figure is the "exceedance
# plot": the actual return series with the model's VaR threshold overlaid,
# and violations (Return < VaR) marked. This is what makes violation
# possible clustering visible: red markers that bunch together provide a
# descriptive pattern that is assessed formally by the independence statistic.

plot_exceedances <- function(data, var_col, title) {
  d <- data[!is.na(data[[var_col]]), ]
  plot(d$Date, d$Return, type = "l", col = "grey50", lwd = 0.7,
       main = title, xlab = "Date", ylab = "Log Return")
  lines(d$Date, d[[var_col]], col = "blue", lwd = 1.5)
  viol <- d$Return < d[[var_col]]
  points(d$Date[viol], d$Return[viol], col = "red", pch = 19, cex = 0.7)
  legend("bottomleft", legend = c("Return", "VaR (99%)", "Violation"),
         col = c("grey50", "blue", "red"), lty = c(1, 1, NA), pch = c(NA, NA, 19),
         lwd = c(0.7, 1.5, NA), bty = "n", cex = 0.8)
}

# Full-sample exceedance plots for the two most-discussed models
plot_exceedances(results_df, "VaR_EVT_99",  "EVT 99% VaR: Forecast vs Actual Returns (Full Sample)")
plot_exceedances(results_df, "VaR_LSTM_99", "LSTM 99% VaR: Forecast vs Actual Returns (Full Sample)")

# Zoomed in on the 2008 GFC window: this is where the LSTM's periodic-refit
# design (a fresh model only every ~250 days) should visibly lag a
# faster-adapting rolling model, and where violation clustering is most
# visually prominent for several models.

gfc_window <- results_df[results_df$Regime == "GFC", ]

plot_exceedances(gfc_window, "VaR_EVT_99",  "EVT 99% VaR During the 2008 GFC")
plot_exceedances(
  gfc_window,
  "VaR_LSTM_99",
  "LSTM 99% VaR During the GFC (Forecasts From 24 Dec 2007)"
)

# Violation-rate comparison across all six models, on the fair common sample
# (Step 8, all-6 intersection), against the 1% nominal target.

barplot(
  height = backtest_all$Violation_Rate,
  names.arg = backtest_all$Model,
  col = "steelblue",
  main = "99% VaR Violation Rate by Model (Common Sample)",
  ylab = "Violation Rate",
  ylim = c(0, max(backtest_all$Violation_Rate) * 1.15),
  cex.names = 0.75,
  las = 1
)
abline(h = 0.01, col = "red", lwd = 2, lty = 2)
legend("topright", legend = "1% nominal target", col = "red", lty = 2, lwd = 2, bty = "n")

# Full-sample mean-excess plot as descriptive context for the 90th-percentile
# POT choice. It cannot validate every rolling threshold: each 127-day window
# has only about 13 exceedances, which explains the unstable GPD fits and must
# be treated as a limitation rather than hidden by the diagnostic.

losses_full <- -na.omit(results_df$Return)
meplot(losses_full, main = "Mean Excess Plot (Full-Sample Losses)")
abline(v = quantile(losses_full, 0.90), col = "red", lty = 2)

### Step 10: Crisis-vs-Calm Backtest by Model (RQ1-3)

# RQ1-3 require comparing all six models across paired calm/crisis windows,
# not just a single Regime label spanning the whole sample. These four
# windows are defined independently of the existing Regime column (which is
# left untouched) and match the dates justified in Step 4:
#   Non-Crisis (2008 pair) - 2004-01-01 to 2006-12-31
#   Crisis     (2008 pair) - 2007-08-01 to 2009-06-30
#   Non-Crisis (COVID pair)- 2017-01-01 to 2019-12-31
#   Crisis     (COVID pair)- 2020-02-01 to 2020-04-30

crisis_windows <- list(
  "Non-Crisis (2004-2006)" = c("2004-01-01", "2006-12-31"),
  "Crisis (2008 GFC)"      = c("2007-08-01", "2009-06-30"),
  "Non-Crisis (2017-2019)" = c("2017-01-01", "2019-12-31"),
  "Crisis (COVID 2020)"    = c("2020-02-01", "2020-04-30")
)

# LSTM's walk-forward forecasts begin on 2007-12-24 after its 1000-day
# burn-in, so it has no coverage in 2004-2006. That window uses a common
# five-model intersection; the other windows use a common six-model
# intersection. Every model within a window is therefore evaluated on exactly
# the same dates.

model_cols_noLSTM <- model_cols[c("Historical", "Gaussian", "Modified", "EGARCH", "EVT")]

crisis_backtest_results <- do.call(rbind, lapply(names(crisis_windows), function(w) {
  rng <- crisis_windows[[w]]
  win_data <- results_df[
    results_df$Date >= as.Date(rng[1]) & results_df$Date <= as.Date(rng[2]),
  ]
  cols <- if (w == "Non-Crisis (2004-2006)") model_cols_noLSTM else model_cols
  common_mask <- complete.cases(win_data[, c("Return", cols)])
  common_window <- win_data[common_mask, ]
  tbl <- common_backtest(common_window, cols)
  tbl$Window <- w
  tbl[, c("Window", setdiff(names(tbl), "Window"))]
}))

cat("\n=== Crisis-vs-Calm Backtest by Model (RQ1-3) ===\n")
print(crisis_backtest_results, row.names = FALSE)

# NOTE ON INTERPRETATION: the COVID crisis window has an expected violation
# count below one at the 1% level, and its common sample may be shorter if a
# model forecast is missing. All three likelihood-ratio tests therefore have
# low power in this window. Report raw counts, exact expected counts and rates
# alongside the formal decisions.

### Step 11: Best-Model Deep Dive Across Crises (Support RQ4 & RQ5)

# Support RQ4 needs a transparent relative definition of "best" because a
# formal rejection is a binary diagnostic, not a model ranking. Within each
# crisis, models are evaluated on identical dates and ranked by the joint
# conditional-coverage LR statistic (lower is better). This criterion reflects
# both frequency and independence. Ranks—not raw LR values—are then averaged
# across crises because the GFC and COVID sample sizes differ substantially.

gfc_tbl   <- crisis_backtest_results[crisis_backtest_results$Window == "Crisis (2008 GFC)", ]
covid_tbl <- crisis_backtest_results[crisis_backtest_results$Window == "Crisis (COVID 2020)", ]

gfc_tbl$Rank   <- rank(gfc_tbl$Conditional_Coverage_LR, na.last = "keep")
covid_tbl$Rank <- rank(covid_tbl$Conditional_Coverage_LR, na.last = "keep")

rank_tbl <- merge(
  gfc_tbl[, c("Model", "Rank")],
  covid_tbl[, c("Model", "Rank")],
  by = "Model", suffixes = c("_GFC", "_COVID")
)
rank_tbl$Avg_Rank <- rowMeans(rank_tbl[, c("Rank_GFC", "Rank_COVID")])
rank_tbl <- rank_tbl[order(rank_tbl$Avg_Rank), ]

cat("\n=== Model Ranking Across Both Crises (conditional-coverage LR) ===\n")
print(rank_tbl, row.names = FALSE)

best_model_name <- rank_tbl$Model[1]
cat("\nBest-performing model across both crises:", best_model_name, "\n")

best_var_col <- model_cols[[best_model_name]]

# Support RQ4: for the best-performing model, compare risk-estimation and
# failure patterns between the 2008 GFC and the COVID shock directly.

best_model_summary <- crisis_backtest_results[
  crisis_backtest_results$Model == best_model_name &
    crisis_backtest_results$Window %in% c("Crisis (2008 GFC)", "Crisis (COVID 2020)"),
]

cat("\n=== Support RQ4:", best_model_name, "- 2008 GFC vs COVID 2020 ===\n")
print(best_model_summary, row.names = FALSE)

gfc_window_best   <- results_df[
  results_df$Date >= as.Date("2007-08-01") & results_df$Date <= as.Date("2009-06-30"),
]
covid_window_best <- results_df[
  results_df$Date >= as.Date("2020-02-01") & results_df$Date <= as.Date("2020-04-30"),
]

# Use the same six-model common dates that underpin the crisis ranking, so the
# best-model summary, plots and persistence diagnostics all refer to one sample.

gfc_window_best <- gfc_window_best[
  complete.cases(gfc_window_best[, c("Return", model_cols)]),
]
covid_window_best <- covid_window_best[
  complete.cases(covid_window_best[, c("Return", model_cols)]),
]

plot_exceedances(
  gfc_window_best, best_var_col,
  paste(best_model_name, "99% VaR During the 2008 GFC (Best Model)")
)
plot_exceedances(
  covid_window_best, best_var_col,
  paste(best_model_name, "99% VaR During the COVID Shock (Best Model)")
)

# Support RQ5: are violations more clustered/persistent during the gradual
# 2008 crisis than during the sudden COVID shock? Two complementary measures
# are used on the best model's violations within each crisis window:
#   (1) the INDEPENDENCE-only component of Christoffersen's test, isolated
#       via LR_ind = LR_cc - LR_uc (Christoffersen, 1998), tested against a
#       chi-squared(1) distribution. This isolates clustering specifically,
#       rather than conflating it with overall coverage as the joint cc
#       p-value does.
#   (2) the longest run of CONSECUTIVE violation days, a simple descriptive
#       measure of how "bunched" the failures are.

### NOTE:
# This helper summarizes violation independence and persistence for a selected
# model and evaluation window. It reuses the standard Christoffersen
# independence result from coverage_tests() and additionally calculates the
# longest run of consecutive violations. Because common-date samples may
# contain gaps, Row_ID is once again used to divide the observations into genuinely
# contiguous segments. Without this step, violations separated by omitted
# trading dates could be incorrectly treated as consecutive, distorting the
# transition counts, independence p-value, and maximum violation run. The
# maximum run is descriptive evidence for RQ5 and is not a separate formal
# backtest.

compute_independence <- function(data, var_col) {
  d <- data[!is.na(data[[var_col]]) & !is.na(data$Return), ]
  bt <- coverage_tests(d, var_col)
  viol <- as.integer(d$Return < d[[var_col]])
  segment_id <- if (nrow(d) == 0) integer(0) else cumsum(c(TRUE, diff(d$Row_ID) != 1))
  max_run <- if (!any(viol == 1)) {
    0
  } else {
    max(vapply(split(viol, segment_id), function(v) {
      runs <- rle(v)
      if (any(runs$values == 1)) max(runs$lengths[runs$values == 1]) else 0
    }, numeric(1)))
  }

  list(
    N = nrow(d),
    Violations = sum(viol),
    LR_ind = bt$LR_IND,
    p_ind = bt$p_IND,
    Max_Consecutive_Violations = max_run
  )
}

gfc_ind   <- compute_independence(gfc_window_best, best_var_col)
covid_ind <- compute_independence(covid_window_best, best_var_col)

persistence_tbl <- data.frame(
  Crisis = c("2008 GFC", "COVID 2020"),
  N = c(gfc_ind$N, covid_ind$N),
  Violations = c(gfc_ind$Violations, covid_ind$Violations),
  LR_independence = c(gfc_ind$LR_ind, covid_ind$LR_ind),
  p_independence = c(gfc_ind$p_ind, covid_ind$p_ind),
  Max_Consecutive_Violations = c(
    gfc_ind$Max_Consecutive_Violations,
    covid_ind$Max_Consecutive_Violations
  )
)

cat("\n=== Violation Clustering/Persistence:", best_model_name, "(Support RQ5) ===\n")
print(persistence_tbl, row.names = FALSE)

# NOTE: a low p_independence (reject H0 of independence) means violations
# are NOT randomly scattered in time but cluster together. Comparing
# LR_independence and Max_Consecutive_Violations across the two crises
# indicates whether the gradual 2008 downturn produced more sustained
# violation streaks than the sudden, sharp COVID shock, or vice versa.

### THIS SECTION FOCUSES ON SAVING CSVs, GRAPHICS etc. (ARTIFACTS) PLUS OTHER THINGS
### THOUGHT OF ALONG THE WAY OF BUILDING THE FINAL DISSERTATION TO UTILIZE THE ENV RAM.

### Step 12: Export Results Artifacts

# Everything up to this point only prints to the console / R graphics
# window. This step writes every results table to CSV, records session
# information in a text file,
# and re-draws every plot used in the results/EDA sections to a PNG file,
# so they can be dropped directly into the dissertation. No upstream logic
# is changed here - plots are simply redrawn into a file device using
# objects already computed earlier in the script, and tables are written
# as-is.

fig_dir <- "outputs/figures"
tbl_dir <- "outputs/tables"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

save_plot <- function(filename, expr, width = 1600, height = 1000, res = 150) {
  png(file.path(fig_dir, filename), width = width, height = height, res = res)
  expr
  dev.off()
}

## Tables

write.csv(VaR_results, file.path(tbl_dir, "01_baseline_VaR_by_confidence.csv"), row.names = FALSE)

# Regime-level Kupiec test summary for the 3 baseline models (Step 3/5),
# rebuilt into a single exportable table from the same violation vectors
# already computed earlier - no re-estimation involved.
baseline_regime_rows <- list()
for (m in c("Historical", "Gaussian", "Modified")) {
  vcol <- switch(m, Historical = "Viol_Hist_99", Gaussian = "Viol_Gauss_99", Modified = "Viol_Mod_99")
  for (reg in c("Calm", "GFC", "COVID")) {
    sub <- subset(results_df, Regime == reg)[[vcol]]
    kt <- kupiec_test(sub)
    baseline_regime_rows[[length(baseline_regime_rows) + 1]] <- data.frame(
      Model = m, Regime = reg,
      N = kt$Observations, Violations = kt$Violations,
      Expected = 0.01 * kt$Observations,
      Violation_Rate = kt$Violation_Rate,
      LR_stat = kt$LR_stat, p_value = kt$p_value
    )
  }
}
baseline_regime_kupiec <- do.call(rbind, baseline_regime_rows)
write.csv(baseline_regime_kupiec, file.path(tbl_dir, "02_baseline_regime_kupiec.csv"), row.names = FALSE)

write.csv(comparison_table, file.path(tbl_dir, "03_comparison_table_full_sample.csv"), row.names = FALSE)
write.csv(backtest_noLSTM, file.path(tbl_dir, "04_common_sample_backtest_no_lstm.csv"), row.names = FALSE)
write.csv(backtest_all, file.path(tbl_dir, "05_common_sample_backtest_all6.csv"), row.names = FALSE)
write.csv(crisis_backtest_results, file.path(tbl_dir, "06_crisis_vs_calm_backtest.csv"), row.names = FALSE)
write.csv(rank_tbl, file.path(tbl_dir, "07_best_model_ranking.csv"), row.names = FALSE)
write.csv(best_model_summary, file.path(tbl_dir, "08_best_model_2008_vs_covid.csv"), row.names = FALSE)
write.csv(persistence_tbl, file.path(tbl_dir, "09_violation_clustering_persistence.csv"), row.names = FALSE)
write.csv(forecast_availability, file.path(tbl_dir, "10_forecast_availability.csv"), row.names = FALSE)

# Record the exact R and package environment used for the run. This is not an
# input dependency; it is provenance that helps diagnose future version drift.
writeLines(capture.output(sessionInfo()), file.path(tbl_dir, "11_session_info.txt"))

## Figures: Step 2 (Return Distribution & Volatility Clustering)

save_plot("01_histogram_log_returns.png", {
  hist(sp500_df$Log_Returns, breaks = 50, probability = TRUE,
       main = "Histogram of S&P 500 Log Returns", xlab = "Log Returns")
  curve(dnorm(x, mean = mean_return, sd = sqrt(var_return)), col = "red", lwd = 2, add = TRUE)
})

save_plot("02_qq_plot_log_returns.png", {
  qqnorm(sp500_df$Log_Returns, main = "QQ Plot of S&P 500 Log Returns",
         xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
  qqline(sp500_df$Log_Returns, col = "red", lwd = 2)
})

save_plot("03_squared_log_returns.png", {
  plot(sp500_df$Date, sp500_df$Squared_Returns, type = "l", col = "black",
       xlab = "Date", ylab = "Squared Log Returns",
       main = "Squared Log Returns of S&P 500 Index")
})

save_plot("04_garch_residuals.png", {
  plot(resid(garch_model), type = "l", col = "black",
       main = "GARCH(1,1) Model Residuals",
       ylab = "Residuals", xlab = "Time")
})

save_plot("05_garch_residuals_qq.png", {
  qqnorm(resid(garch_model))
  qqline(resid(garch_model), col = "red")
})

save_plot("06_garch_conditional_variance.png", {
  plot(fitted(garch_model)[,1]^2, type = "l", col = "blue",
       main = "Conditional Variance from GARCH(1,1)",
       ylab = "Estimated Conditional Variance", xlab = "Time")
})

save_plot("07_acf_returns.png", {
  acf(sp500_df$Log_Returns, main = "ACF of Returns")
})

save_plot("08_acf_squared_returns.png", {
  acf(sp500_df$Log_Returns^2, main = "ACF of Squared Returns")
})

## Figures: Step 4 (Crisis Window Justification)

save_plot("09_rolling_vol_full_sample.png", {
  plot(sp500_df$Date, sp500_df$RollingVol20, type = "l",
       main = "20-Day Rolling Volatility of S&P 500 Returns",
       xlab = "Date", ylab = "Rolling Volatility")
  abline(v = as.Date("2020-02-01"), col = "red", lwd = 2)
  abline(v = as.Date("2020-04-30"), col = "red", lwd = 2)
})

save_plot("10_rolling_vol_covid_zoom.png", {
  covid_start <- as.Date("2019-09-01")
  covid_end <- as.Date("2020-06-30")
  subset_covid <- sp500_df[sp500_df$Date >= covid_start & sp500_df$Date <= covid_end, ]
  plot(subset_covid$Date, subset_covid$RollingVol20, type = "l", lwd = 2,
       main = "20-Day Rolling Volatility Around the COVID Shock",
       xlab = "Date", ylab = "Rolling Volatility")
  abline(v = as.Date("2020-02-01"), col = "red", lwd = 2)
  abline(v = as.Date("2020-04-30"), col = "red", lwd = 2)
})

save_plot("11_rolling_vol_2008_zoom.png", {
  crisis2008_start <- as.Date("2006-01-01")
  crisis2008_end   <- as.Date("2010-12-31")
  subset_2008 <- sp500_df[sp500_df$Date >= crisis2008_start & sp500_df$Date <= crisis2008_end, ]
  plot(subset_2008$Date, subset_2008$RollingVol20, type = "l", lwd = 2,
       main = "20-Day Rolling Volatility Around the 2008 Financial Crisis",
       xlab = "Date", ylab = "Rolling Volatility")
  abline(v = as.Date("2007-08-01"), col = "red", lwd = 2)
  abline(v = as.Date("2009-06-30"), col = "red", lwd = 2)
})

## Figures: Step 9 (Results Visualizations)

save_plot("12_evt_exceedance_full_sample.png", {
  plot_exceedances(results_df, "VaR_EVT_99", "EVT 99% VaR: Forecast vs Actual Returns (Full Sample)")
})

save_plot("13_lstm_exceedance_full_sample.png", {
  plot_exceedances(results_df, "VaR_LSTM_99", "LSTM 99% VaR: Forecast vs Actual Returns (Full Sample)")
})

save_plot("14_evt_exceedance_gfc_zoom.png", {
  plot_exceedances(gfc_window, "VaR_EVT_99", "EVT 99% VaR During the 2008 GFC")
})

save_plot("15_lstm_exceedance_gfc_zoom.png", {
  plot_exceedances(
    gfc_window,
    "VaR_LSTM_99",
    "LSTM 99% VaR During the GFC (Forecasts From 24 Dec 2007)"
  )
})

save_plot("16_violation_rate_barchart.png", {
  barplot(
    height = backtest_all$Violation_Rate,
    names.arg = backtest_all$Model,
    col = "steelblue",
    main = "99% VaR Violation Rate by Model (Common Sample)",
    ylab = "Violation Rate",
    ylim = c(0, max(backtest_all$Violation_Rate) * 1.15),
    cex.names = 0.75,
    las = 1
  )
  abline(h = 0.01, col = "red", lwd = 2, lty = 2)
  legend("topright", legend = "1% nominal target", col = "red", lty = 2, lwd = 2, bty = "n")
})

save_plot("17_mean_excess_plot.png", {
  meplot(losses_full, main = "Mean Excess Plot (Full-Sample Losses)")
  abline(v = quantile(losses_full, 0.90), col = "red", lty = 2)
})

## Figures: Step 11 (Best-Model Deep Dive)

save_plot("18_best_model_gfc_exceedance.png", {
  plot_exceedances(
    gfc_window_best, best_var_col,
    paste(best_model_name, "99% VaR During the 2008 GFC (Best Model)")
  )
})

save_plot("19_best_model_covid_exceedance.png", {
  plot_exceedances(
    covid_window_best, best_var_col,
    paste(best_model_name, "99% VaR During the COVID Shock (Best Model)")
  )
})

## Figures: Calm-vs-Crisis Violation Rates

# The calm chart uses 2017-2019 because it is the only predefined calm window
# with forecasts from all six models; the LSTM had not begun forecasting in
# 2004-2006. The crisis chart pools the GFC and COVID common samples by summing
# violations and observations for each model. This weights each crisis by its
# number of evaluated dates and preserves like-for-like model comparisons.
model_order <- names(model_cols)

calm_plot_data <- crisis_backtest_results[
  crisis_backtest_results$Window == "Non-Crisis (2017-2019)",
  c("Model", "N", "Violations", "Violation_Rate")
]
calm_plot_data <- calm_plot_data[match(model_order, calm_plot_data$Model), ]

crisis_plot_source <- crisis_backtest_results[
  crisis_backtest_results$Window %in% c("Crisis (2008 GFC)", "Crisis (COVID 2020)"),
  c("Model", "N", "Violations")
]
crisis_plot_data <- aggregate(
  cbind(N, Violations) ~ Model,
  data = crisis_plot_source,
  FUN = sum
)
crisis_plot_data$Violation_Rate <- crisis_plot_data$Violations / crisis_plot_data$N
crisis_plot_data <- crisis_plot_data[match(model_order, crisis_plot_data$Model), ]

# Use a compact 0-5% scale for the calm-market chart so differences among the
# relatively low rates remain readable. The crisis chart requires a wider
# scale because its violation rates are substantially higher. Exact percentage
# labels are retained on both figures to keep the different scales explicit.
calm_plot_ylim <- c(0, 0.05)
crisis_plot_ylim <- c(0, 1.22 * max(crisis_plot_data$Violation_Rate))

plot_regime_violation_rates <- function(data, title, subtitle, y_limits) {
  par(
    mfrow = c(1, 1),
    fig = c(0, 1, 0, 1),
    oma = c(0, 0, 0, 0),
    mar = c(6.2, 5.2, 4.2, 2.0) + 0.1
  )
  bar_positions <- barplot(
    height = data$Violation_Rate,
    names.arg = data$Model,
    col = "steelblue",
    main = title,
    sub = subtitle,
    ylab = "Violation Rate",
    ylim = y_limits,
    cex.names = 0.75,
    las = 1
  )
  abline(h = 0.01, col = "red", lwd = 2, lty = 2)
  text(
    x = bar_positions,
    y = data$Violation_Rate,
    labels = sprintf("%.2f%%", 100 * data$Violation_Rate),
    pos = 3,
    cex = 0.72
  )
  legend(
    "topright", legend = "1% nominal target",
    col = "red", lty = 2, lwd = 2, bty = "n"
  )
}

save_plot("20_violation_rate_calm_common_sample.png", {
  plot_regime_violation_rates(
    calm_plot_data,
    "99% VaR Violation Rate by Model: Calm Market",
    "2017-2019 six-model common sample (n = 528 per model)",
    calm_plot_ylim
  )
})

save_plot("21_violation_rate_crisis_common_sample.png", {
  plot_regime_violation_rates(
    crisis_plot_data,
    "99% VaR Violation Rate by Model: Financial Crises",
    "Pooled GFC and COVID common samples (n = 177 per model)",
    crisis_plot_ylim
  )
})

## Figure 22: GFC-versus-COVID Model Comparison (Support RQ3)

# This final block can be rerun on its own after the analysis above has created
# crisis_backtest_results, model_order, save_plot and fig_dir. Unlike Figure 21,
# it keeps the two crises separate so that regime-dependent performance remains
# visible. Each crisis still uses identical evaluation dates across all models.
rq3_plot_data <- crisis_backtest_results[
  crisis_backtest_results$Window %in% c(
    "Crisis (2008 GFC)", "Crisis (COVID 2020)"
  ),
  c("Window", "Model", "N", "Violation_Rate")
]

gfc_rq3 <- rq3_plot_data[rq3_plot_data$Window == "Crisis (2008 GFC)", ]
covid_rq3 <- rq3_plot_data[rq3_plot_data$Window == "Crisis (COVID 2020)", ]
gfc_rq3 <- gfc_rq3[match(model_order, gfc_rq3$Model), ]
covid_rq3 <- covid_rq3[match(model_order, covid_rq3$Model), ]

stopifnot(
  identical(gfc_rq3$Model, model_order),
  identical(covid_rq3$Model, model_order),
  length(unique(gfc_rq3$N)) == 1L,
  length(unique(covid_rq3$N)) == 1L
)

# Rows define the two coloured crisis bars and columns define model groups.
rq3_rate_matrix <- rbind(
  `2008 GFC` = gfc_rq3$Violation_Rate, # `` non-standard variable name.
  `COVID-19` = covid_rq3$Violation_Rate
)
colnames(rq3_rate_matrix) <- model_order

save_plot("22_violation_rate_gfc_vs_covid.png", {
  par(
    mfrow = c(1, 1),
    fig = c(0, 1, 0, 1),
    oma = c(0, 0, 0, 0),
    mar = c(6.2, 5.2, 4.7, 2.0) + 0.1
  )
  rq3_bar_positions <- barplot(
    height = rq3_rate_matrix,
    beside = TRUE,
    names.arg = model_order,
    col = c("steelblue", "darkorange"),
    border = "black",
    main = "99% VaR Violation Rates: GFC versus COVID-19",
    sub = sprintf(
      "Six-model common samples: GFC n = %d; COVID-19 n = %d",
      unique(gfc_rq3$N), unique(covid_rq3$N)
    ),
    ylab = "Violation Rate",
    ylim = c(0, 1.20 * max(rq3_rate_matrix)),
    cex.names = 0.75,
    las = 1
  )
  abline(h = 0.01, col = "red", lwd = 2, lty = 2)
  text(
    x = rq3_bar_positions,
    y = rq3_rate_matrix,
    labels = sprintf("%.2f%%", 100 * rq3_rate_matrix),
    pos = 3,
    cex = 0.68
  )
  graphics::legend(
    "topleft",
    legend = c("2008 GFC", "COVID-19", "1% nominal target"),
    col = c("steelblue", "darkorange", "red"),
    pch = c(15, 15, NA_integer_),
    lty = c(NA, NA, 2),
    lwd = c(NA, NA, 2),
    pt.cex = 1.2,
    bty = "n"
  )
})

## EGARCH-Only Full-Crisis Robustness Check (Support RQ4 and RQ5)

# The primary RQ4/RQ5 results above retain the six-model common dates used to
# select EGARCH fairly. This additional check does not replace them. Once the
# analysis focuses only on EGARCH, it removes the EVT-driven common-date
# restriction and uses every valid EGARCH forecast inside the same predefined
# crisis windows. No crisis boundary or model specification is changed.
egarch_full_windows <- list(
  `2008 GFC` = c(as.Date("2007-08-01"), as.Date("2009-06-30")),
  `COVID-19` = c(as.Date("2020-02-01"), as.Date("2020-04-30"))
)

egarch_full_window_data <- lapply(egarch_full_windows, function(bounds) {
  d <- results_df[
    results_df$Date >= bounds[1] & results_df$Date <= bounds[2] &
      !is.na(results_df$Return) & !is.na(results_df$VaR_EGARCH_99),
  ]
  d[order(d$Row_ID), ]
})

egarch_robustness_row <- function(d, crisis, alpha = 0.01) {
  bt <- coverage_tests(d, "VaR_EGARCH_99", alpha)
  violations <- as.integer(d$Return < d$VaR_EGARCH_99)
  exceedance_magnitude <- d$VaR_EGARCH_99[violations == 1] -
    d$Return[violations == 1]

  # Transition counts exclude pairs separated by unavailable dates. EGARCH is
  # expected to be contiguous here, but retaining this guard keeps the
  # robustness calculation consistent with the primary backtests.
  if (length(violations) >= 2L) {
    contiguous_pair <- diff(d$Row_ID) == 1L
    previous <- violations[-length(violations)][contiguous_pair]
    current <- violations[-1L][contiguous_pair]
    n00 <- sum(previous == 0L & current == 0L)
    n01 <- sum(previous == 0L & current == 1L)
    n10 <- sum(previous == 1L & current == 0L)
    n11 <- sum(previous == 1L & current == 1L)
  } else {
    n00 <- n01 <- n10 <- n11 <- 0L
  }

  segment_id <- if (nrow(d) == 0L) {
    integer(0)
  } else {
    cumsum(c(TRUE, diff(d$Row_ID) != 1L))
  }
  violation_runs <- unlist(lapply(split(violations, segment_id), function(v) {
    runs <- rle(v)
    runs$lengths[runs$values == 1L]
  }), use.names = FALSE)
  max_run <- if (length(violation_runs) == 0L) 0L else max(violation_runs)
  n_clusters <- length(violation_runs)

  inter_violation_gaps <- unlist(lapply(split(d$Row_ID[violations == 1L],
                                                  segment_id[violations == 1L]),
                                          function(ids) {
    if (length(ids) >= 2L) diff(ids) else numeric(0)
  }), use.names = FALSE)

  safe_summary <- function(x, fun) {
    if (length(x) == 0L) NA_real_ else fun(x)
  }

  data.frame(
    Crisis = crisis,
    Start_Date = min(d$Date),
    End_Date = max(d$Date),
    N = bt$N,
    Violations = bt$Violations,
    Expected = bt$Expected,
    Violation_Rate = bt$Violation_Rate,
    Mean_Return = mean(d$Return),
    Minimum_Return = min(d$Return),
    Mean_VaR = mean(d$VaR_EGARCH_99),
    Minimum_VaR = min(d$VaR_EGARCH_99),
    Mean_Exceedance = safe_summary(exceedance_magnitude, mean),
    Median_Exceedance = safe_summary(exceedance_magnitude, median),
    Maximum_Exceedance = safe_summary(exceedance_magnitude, max),
    Kupiec_LR = bt$LR_UC,
    Kupiec_p = bt$p_UC,
    Kupiec = bt$Decision_UC,
    Independence_LR = bt$LR_IND,
    Independence_p = bt$p_IND,
    Independence = bt$Decision_IND,
    Conditional_Coverage_LR = bt$LR_CC,
    Conditional_Coverage_p = bt$p_CC,
    Conditional_Coverage = bt$Decision_CC,
    n00 = n00,
    n01 = n01,
    n10 = n10,
    n11 = n11,
    Violation_Clusters = n_clusters,
    Max_Consecutive_Violations = max_run,
    Mean_Inter_Violation_Gap = safe_summary(inter_violation_gaps, mean),
    Median_Inter_Violation_Gap = safe_summary(inter_violation_gaps, median),
    stringsAsFactors = FALSE
  )
}

egarch_full_robustness <- do.call(rbind, lapply(
  names(egarch_full_window_data),
  function(crisis) {
    egarch_robustness_row(egarch_full_window_data[[crisis]], crisis)
  }
))
rownames(egarch_full_robustness) <- NULL

cat("\n=== EGARCH Full-Crisis Robustness Check (RQ4/RQ5) ===\n")
print(egarch_full_robustness, row.names = FALSE)

write.csv(
  egarch_full_robustness,
  file.path(tbl_dir, "12_egarch_full_window_robustness.csv"),
  row.names = FALSE
)

# A compact sensitivity table makes the change in sample construction explicit
# without displacing the primary common-sample results.
egarch_common_sensitivity <- data.frame(
  Crisis = c("2008 GFC", "COVID-19"),
  Sample = "Six-model common dates",
  N = best_model_summary$N,
  Violations = best_model_summary$Violations,
  Expected = best_model_summary$Expected,
  Violation_Rate = best_model_summary$Violation_Rate,
  Kupiec_p = best_model_summary$Kupiec_p,
  Independence_p = best_model_summary$Independence_p,
  Conditional_Coverage_p = best_model_summary$Conditional_Coverage_p,
  stringsAsFactors = FALSE
)

egarch_full_sensitivity <- data.frame(
  Crisis = egarch_full_robustness$Crisis,
  Sample = "All valid EGARCH dates",
  N = egarch_full_robustness$N,
  Violations = egarch_full_robustness$Violations,
  Expected = egarch_full_robustness$Expected,
  Violation_Rate = egarch_full_robustness$Violation_Rate,
  Kupiec_p = egarch_full_robustness$Kupiec_p,
  Independence_p = egarch_full_robustness$Independence_p,
  Conditional_Coverage_p = egarch_full_robustness$Conditional_Coverage_p,
  stringsAsFactors = FALSE
)

egarch_sample_sensitivity <- rbind(
  egarch_common_sensitivity,
  egarch_full_sensitivity
)
egarch_sample_sensitivity <- egarch_sample_sensitivity[order(
  egarch_sample_sensitivity$Crisis,
  factor(
    egarch_sample_sensitivity$Sample,
    levels = c("Six-model common dates", "All valid EGARCH dates")
  )
), ]
rownames(egarch_sample_sensitivity) <- NULL

write.csv(
  egarch_sample_sensitivity,
  file.path(tbl_dir, "13_egarch_common_vs_full_sensitivity.csv"),
  row.names = FALSE
)

plot_egarch_full_window <- function(d, title, subtitle) {
  par(
    mfrow = c(1, 1),
    fig = c(0, 1, 0, 1),
    oma = c(0, 0, 0, 0),
    mar = c(5.2, 5.2, 4.5, 2.0) + 0.1
  )
  violations <- d$Return < d$VaR_EGARCH_99
  plot(
    d$Date, d$Return,
    type = "l", col = "grey50", lwd = 0.7,
    main = title, sub = subtitle,
    xlab = "Date", ylab = "Log Return",
    ylim = range(c(d$Return, d$VaR_EGARCH_99), na.rm = TRUE)
  )
  lines(d$Date, d$VaR_EGARCH_99, col = "blue", lwd = 1.5)
  points(
    d$Date[violations], d$Return[violations],
    col = "red", pch = 19, cex = 0.8
  )
  graphics::legend(
    "bottomleft",
    legend = c("Return", "EGARCH VaR (99%)", "Violation"),
    col = c("grey50", "blue", "red"),
    lty = c(1, 1, NA),
    pch = c(NA, NA, 19),
    lwd = c(0.7, 1.5, NA),
    bty = "n", cex = 0.8
  )
}

save_plot("23_egarch_full_gfc_robustness.png", {
  plot_egarch_full_window(
    egarch_full_window_data[["2008 GFC"]],
    "EGARCH 99% VaR During the Full GFC Window",
    sprintf(
      "All valid EGARCH dates (n = %d)",
      nrow(egarch_full_window_data[["2008 GFC"]])
    )
  )
})

save_plot("24_egarch_full_covid_robustness.png", {
  plot_egarch_full_window(
    egarch_full_window_data[["COVID-19"]],
    "EGARCH 99% VaR During the Full COVID-19 Window",
    sprintf(
      "All valid EGARCH dates (n = %d)",
      nrow(egarch_full_window_data[["COVID-19"]])
    )
  )
})

## Appendix Figure: LSTM During the COVID-19 Shock

# The GFC LSTM plot above shows the model's weakest crisis result. This
# matching COVID plot documents the contrasting period in which the LSTM had
# the lowest violation rate of the six models. It is supplementary evidence
# for the regime sensitivity discussed in RQ3, rather than another ranking
# figure for the main Results section.
lstm_covid_appendix <- results_df[
  results_df$Date >= as.Date("2020-02-01") &
    results_df$Date <= as.Date("2020-04-30"),
]

save_plot("25_lstm_exceedance_covid_appendix.png", {
  plot_exceedances(
    lstm_covid_appendix,
    "VaR_LSTM_99",
    "LSTM 99% VaR During the COVID-19 Shock"
  )
})

cat("\n=== Artifacts exported ===\n")
cat("Table/provenance files:", length(list.files(tbl_dir)), "files in", tbl_dir, "\n")
cat("Figures:", length(list.files(fig_dir)), "PNG files in", fig_dir, "\n")
