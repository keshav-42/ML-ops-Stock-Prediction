/** Human-readable names for the model's feature columns (see data_spec.md). */

export const FEATURE_LABEL: Record<string, string> = {
  ret_1: "1-day return",
  ret_5: "5-day return",
  ret_10: "10-day return",
  ret_21: "21-day return",
  rstd_5: "5-day realized vol",
  rstd_10: "10-day realized vol",
  rstd_21: "21-day realized vol",
  parkinson_t: "Parkinson range vol (today)",
  gk_t: "Garman–Klass vol (today)",
  gk_ma_5: "GK vol · 5-day avg",
  gk_ma_10: "GK vol · 10-day avg",
  gk_ma_21: "GK vol · 21-day avg",
  atr_14: "Average true range (14d)",
  rsi_14: "RSI (14d)",
  macd: "MACD",
  macd_signal: "MACD signal",
  macd_hist: "MACD histogram",
  vol_z_21: "Volume z-score (21d)",
  nifty_ret_1: "NIFTY 1-day return",
  nifty_rstd_21: "NIFTY 21-day vol",
  vix_level: "India VIX level",
  vix_chg: "India VIX change",
  dow_sin: "Day-of-week (cyclic)",
  dow_cos: "Day-of-week (cyclic)",
  month_sin: "Month (cyclic)",
  month_cos: "Month (cyclic)",
  expiry_week_flag: "F&O expiry week",
  days_to_expiry: "Days to F&O expiry",
};

export function featureLabel(feature: string): string {
  return FEATURE_LABEL[feature] ?? feature;
}
