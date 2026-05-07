# Feature Dictionary

This document describes all features extracted from IntelliCage activity time series, organized by category. Features are computed per animal over the full recording period and used as inputs for unsupervised dominance ranking.

**Total features**: 14 basic + 66 temporal = **80 features** (plus `actual_rank`).

---

## 1. Basic Activity Features (`generate_features`)

These features summarize the raw distribution of activity values across the entire time series.

| Feature | Description |
|---|---|
| `total_activity` | Sum of all activity values over the recording period. |
| `mean_activity` | Average activity per time bin. |
| `std_activity` | Standard deviation of activity — overall variability. |
| `max_activity` | Peak activity value at any single time point. |
| `min_activity` | Minimum activity value observed. |
| `cv_activity` | Coefficient of variation (std / mean). Higher = more irregular. |
| `median_activity` | Median activity; robust to outliers. |
| `max_median_ratio` | Ratio of max to median — how extreme the peak is relative to typical activity. |
| `activity_per_hour` | Average activity per hour (total / 24). |
| `activity_skew` | Skewness of the activity distribution. Positive = rare high bursts. |
| `activity_kurtosis` | Excess kurtosis (Fisher). Measures the "tailedness" of the activity distribution relative to a normal distribution. Kurtosis = 0 means normal-like; **positive** = heavy tails (rare but extreme activity bursts dominate); **negative** = light tails (activity is more uniform, fewer extremes). |
| `gini_coefficient` | Inequality of activity distribution, borrowed from economics. Computed as the area between the Lorenz curve and the line of perfect equality. **0** = all time bins have identical activity (perfectly uniform); **1** = all activity is concentrated in a single bin (maximally bursty). Higher values suggest the animal has intense but rare activity episodes rather than steady continuous movement. |
| `high_activity_frac` | Fraction of time bins above the 75th percentile. |

---

## 2. Day / Night Activity Features (`generate_temporal_features`)

Based on a light/dark cycle where **daytime = 08:00–20:00** and **nighttime = 20:00–08:00** (ZT0 = 20:00).

| Feature | Description |
|---|---|
| `day_activity` | Total activity during daytime hours (08:00–20:00). |
| `night_activity` | Total activity during nighttime hours (20:00–08:00). |
| `day_night_ratio` | Ratio of day to night activity. |
| `day_mean_activity` | Mean activity per bin during daytime. |
| `night_mean_activity` | Mean activity per bin during nighttime. |
| `day_cv_activity` | Coefficient of variation of activity during the day. |
| `night_cv_activity` | Coefficient of variation of activity during the night. |
| `daynight_log_ratio` | Log-scale day/night ratio: log(1+day) − log(1+night). More stable at large differences. |
| `day_frac` | Fraction of total activity occurring during the day. |
| `night_frac` | Fraction of total activity occurring at night. |
| `day_minus_night` | Absolute difference between day and night activity totals. |
| `active_fraction_day` | Fraction of daytime bins where activity exceeds 50% of the animal's mean. |
| `active_fraction_night` | Fraction of nighttime bins above the 50%-of-mean threshold. |

---

## 3. Timing / Onset Features

| Feature | Description |
|---|---|
| `peak_hour` | Hour of the day (0–23) with the highest mean activity. |
| `activity_onset` | Earliest hour where mean hourly activity exceeds 10% of the daily maximum. Approximates wake time. |
| `activity_offset` | Latest hour where activity is still above the 10% threshold. Approximates rest time. |
| `active_window_len` | Length of the active window in hours (offset − onset). |
| `hourly_entropy` | Shannon entropy of the hourly activity distribution. High = spread across all hours; low = concentrated. |
| `hourly_entropy_norm` | Entropy normalized by log(24), ranging 0–1. |
| `peak_to_mean_ratio` | Ratio of peak hourly activity to mean hourly activity. High = sharp, concentrated peak. |

---

## 4. Circadian Rhythm Features

These features capture the strength and shape of the ~24-hour biological rhythm.

### 4a. Spectral (Fourier-based)

Spectral power is computed from the FFT of the mean-centered activity signal. Power at each period is $|F_k|^2$ at the frequency bin closest to the target.

| Feature | Description |
|---|---|
| `power_24h` | Spectral power at the 24-hour frequency. High = strong daily rhythm. |
| `power_12h` | Spectral power at the 12-hour frequency (first harmonic). |
| `power_8h` | Spectral power at the 8-hour frequency (second harmonic). |
| `rhythm_ratio_24_over_harm` | 24h power / (12h + 8h power). High = clean 24h cycle dominates over sub-harmonics. |

### 4b. Cosinor Model

A cosine of the form $A \cos(2\pi t / 24 + \phi) + M$ is fitted via least squares.

| Feature | Description |
|---|---|
| `cosinor_mesor` | MESOR: mean level of the fitted cosine (average rhythm baseline). |
| `cosinor_amplitude` | Amplitude: $\sqrt{\beta_c^2 + \beta_s^2}$, half the peak-to-trough range. |
| `cosinor_acrophase` | Acrophase in radians (0–2π): time within the 24h cycle when the fitted cosine peaks. |

### 4c. Non-Parametric Rhythm Metrics

| Feature | Description |
|---|---|
| `relative_amplitude` | $(M10 - L5) / (M10 + L5)$. M10 is the mean activity of the **10 most active consecutive hours** (rolling 1h bins); L5 is the mean of the **5 least active consecutive hours**. Ranges 0–1. A value near **1** means the animal has a very strong rest/activity contrast (very active during its peak window, nearly silent during its trough). A value near **0** means activity is nearly flat across the day — no clear active or rest phase. Sensitive to disrupted or fragmented rhythms. |
| `interdaily_stability` | IS: regularity of the daily rhythm **across different days**. Computed as $\frac{n \sum \bar{h}_i^2}{p \sum (x_i - \bar{x})^2}$, where $\bar{h}_i$ is the grand mean for each hour of the day (averaged across all days) and $x_i$ are all individual observations. Ranges 0–1. **High IS** (→1) means the animal does the same things at the same times every day — a very stable, predictable rhythm. **Low IS** (→0) means the daily pattern shifts from day to day — irregular or drifting rhythms. Useful for detecting social jet-lag or stress-induced schedule disruption. |
| `intradaily_variability` | IV: **fragmentation** of the activity rhythm within each day. Computed as $\frac{n \sum d_i^2}{(n-1) \sum (x_i - \bar{x})^2}$, where $d_i = x_{i+1} - x_i$ are consecutive differences. IV quantifies how abruptly and frequently the activity signal switches between high and low values. **Low IV** (→0) means the animal has long, sustained bouts of activity and rest with smooth transitions. **High IV** (→2 or above) means activity is highly fragmented — many rapid switches between active and inactive states, like a restless or anxious animal. IV and IS are complementary: an animal can have high IS (same pattern every day) but high IV (that pattern is itself fragmented). |

### 4d. Autocorrelation

Lag is computed in samples based on the inferred sampling interval.

| Feature | Description |
|---|---|
| `autocorr_24h` | Pearson autocorrelation at a 24-hour lag. High = strongly repeating daily pattern. |
| `autocorr_12h` | Autocorrelation at a 12-hour lag. Captures semi-diurnal periodicity. |

---

## 5. Bout Structure Features

A **bout** is a continuous run of time bins where activity exceeds 50% of the animal's mean. A **gap** (inter-bout interval, IBI) is the inactive run between bouts.

### 5a. Bout counts and rates

| Feature | Description |
|---|---|
| `num_bouts` | Total number of activity bouts over the recording. |
| `num_bouts_day` | Number of bouts starting during daytime. |
| `num_bouts_night` | Number of bouts starting during nighttime. |
| `bout_rate_day` | Bouts per hour during the day (bouts / 12). |
| `bout_rate_night` | Bouts per hour during the night (bouts / 12). |
| `bout_rate_ratio` | Day-to-night ratio of bout rate. |
| `transitions_per_hour` | Number of active↔inactive transitions per hour. High = highly fragmented activity. |
| `activity_per_bout` | Total activity divided by number of bouts. Average energy per episode. |

### 5b. Bout length statistics

Lengths are expressed in **time bins** (bin size = `freq_minutes`).

| Feature | Description |
|---|---|
| `mean_bout_len` | Mean duration of activity bouts. |
| `median_bout_len` | Median bout duration — robust to a few very long bouts. |
| `p95_bout_len` | 95th percentile of bout lengths. |
| `bout_len_cv` | Coefficient of variation of bout lengths. High = irregular bout durations. |
| `bout_len_skew` | Skewness of bout length distribution. |
| `bout_len_kurt` | Kurtosis of bout length distribution (Fisher). Positive = a few extremely long bouts dominate (heavy tail); negative = bouts are more uniformly distributed in length. |
| `short_bout_frac` | Fraction of bouts lasting ≤ 1 hour. High = mostly fleeting activity episodes. |
| `longest_active_run` | Length of the single longest continuous activity bout. |

### 5c. Inter-Bout Interval (IBI / gap) statistics

| Feature | Description |
|---|---|
| `ibi_mean` | Mean gap length between bouts (in time bins). |
| `ibi_median` | Median gap length. |
| `ibi_min` | Shortest gap observed. |
| `ibi_max` | Longest gap (longest rest period). |
| `ibi_p25` | 25th percentile of gap lengths. |
| `ibi_p75` | 75th percentile of gap lengths. |
| `ibi_iqr` | IQR of gap lengths (P75 − P25). |
| `ibi_p90` | 90th percentile of gap lengths. |
| `ibi_cv` | Coefficient of variation of gap lengths. |
| `ibi_skew` | Skewness of IBI distribution. |
| `ibi_kurt` | Kurtosis of IBI distribution (Fisher). Positive = a few extremely long rest periods dominate; negative = rest intervals are more uniformly distributed. |
| `short_gap_frac` | Fraction of gaps lasting ≤ 1 hour. High = animal rarely rests for long. |
| `longest_inactive_run` | Length of the single longest continuous inactive period. |

---

## 6. Daily Stability & Burstiness Features

| Feature | Description |
|---|---|
| `daily_total_cv` | CV of total daily activity across recording days. Low = consistent; high = some days much more active. |
| `daily_peak_std` | Standard deviation of the peak activity hour across days. Low = the animal always peaks at the same time. |
| `fano_30m` | Fano factor of activity in 30-minute windows (variance / mean). Fano = 1 for Poisson; higher = bursty. |
| `rolling_std_1h` | Mean of the rolling 1-hour standard deviation of activity. Measures local within-hour noisiness. |

---

## Notes

- **Bout/gap threshold**: 50% of each animal's own mean activity (animal-specific).
- **ZT convention**: ZT0 = lights on at 20:00 local time. Dark phase (when mice are most active) = 20:00–08:00.
- **Sampling interval** (`freq_minutes`): inferred from the data index median diff; defaults to 60 min if indeterminate.
- **All features are per-animal**, one row per animal in the output DataFrame.
- **Scaling**: configurable via `get_data_scaled(scaler_type=...)`. Options: `'standard'` (z-score), `'minmax'` ([0,1]), `'robust'` (median/IQR), `'maxabs'` ([-1,1]).
