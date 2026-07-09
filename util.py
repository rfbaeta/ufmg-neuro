from datetime import datetime
from circadipy import chrono_reader as chr  
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


def get_data_scaled(all_features, feature_cols, use_scale=True, scaler_type='standard'):
    """
    Scale feature matrix X.

    Parameters
    ----------
    scaler_type : str
        'standard'  — StandardScaler: zero mean, unit variance.
                      Each feature: z = (x - mean) / std.
                      Best for algorithms that assume Gaussian distribution (Ridge, PCA, SVM).
                      Sensitive to outliers.

        'minmax'    — MinMaxScaler: scales each feature to [0, 1].
                      z = (x - min) / (max - min).
                      Preserves zero values. Sensitive to outliers.

        'robust'    — RobustScaler: uses median and IQR instead of mean/std.
                      z = (x - median) / IQR.
                      Best choice when data has outliers or non-Gaussian distributions.

        'maxabs'    — MaxAbsScaler: scales by the maximum absolute value to [-1, 1].
                      Does not shift/center data (preserves sparsity).

        None / False / use_scale=False — returns raw unscaled X.
    """
    X = all_features[feature_cols].values
    if not use_scale:
        return X

    scalers = {
        'standard': StandardScaler(),
        'minmax':   MinMaxScaler(),
        'robust':   RobustScaler(),
        'maxabs':   MaxAbsScaler(),
    }

    if scaler_type not in scalers:
        raise ValueError(f"Unknown scaler_type '{scaler_type}'. Choose from: {list(scalers.keys())}")

    scaler = scalers[scaler_type]
    return scaler.fit_transform(X)



def get_intellicage_start_end_date(file):
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    file.close()

    date_type = "%Y-%m-%d %H:%M:%S"        
    start_date = datetime.strptime(lines[1].split("Start date: ", 1)[1].strip(), date_type)
    end_date = datetime.strptime(lines[2].split("End date: ", 1)[1].strip(), date_type)
    #print(f"Start date {start_date} end date {end_date}")
    
    # Calculate number of days
    num_days = (end_date.date() - start_date.date()).days + 1
    
    return num_days
       



# Split animal_3 protocol into per-day dataframes
def split_animal_protocol_by_day(animals_protocols, animal):
    animal_df = animals_protocols[animal].data.copy()
    animal_df['date'] = animal_df.index.normalize()

    # Build a dict of day -> dataframe
    animal_by_day = {
        d: animal_df[animal_df['date'] == d].drop(columns='date')
        for d in sorted(animal_df['date'].unique())
    }
    ks = list(animal_by_day.keys())
    print(f"Animal {animal} Days found:", len(ks))
    return animal_by_day



def read_data(file, labels_dict, name="", zt_0_time = None, type = 'intellicage'):

    protocol = None

    try:
        #print(f"Reading file {file}") 
        num_days = get_intellicage_start_end_date(file)
        #print(f"Number of days: {num_days}")
        labels_dict['cycle_days'] = [num_days+1]
        protocol = chr.read_protocol(name, file, zt_0_time = zt_0_time, labels_dict = labels_dict, type = type, consider_first_day = True)
    except Exception as e:
        print(f"Error reading file {file}: {e}")

    return protocol





def generate_features(animals_protocols, output_path=None):


    animal_features = {}

    for animal_number in animals_protocols.keys():

        
        data = animals_protocols[animal_number].data['values']
        
        total_activity = data.sum()
        sorted_vals = np.sort(data.values)
        n_vals = len(sorted_vals)
        gini = (2.0 * np.sum((np.arange(1, n_vals + 1)) * sorted_vals) / (n_vals * sorted_vals.sum() + 1e-9) - (n_vals + 1) / n_vals) if n_vals > 0 else 0.0
        p75_thresh = data.quantile(0.75)
        high_activity_frac = float((data > p75_thresh).mean())
        features = {
            'animal': animal_number,
            'total_activity': total_activity,
            'mean_activity': data.mean(),
            'std_activity': data.std(),
            'max_activity': data.max(),
            'min_activity': data.min(),
            'cv_activity': data.std() / data.mean() if data.mean() > 0 else 0,
            'median_activity': data.median(),
            'max_median_ratio': data.max() / (data.median() + 1e-9),
            'activity_per_hour': total_activity / 24.0,
            'activity_skew': float(stats.skew(data.values)),
            'activity_kurtosis': float(stats.kurtosis(data.values, fisher=True, bias=False)),
            'gini_coefficient': float(gini),
            'high_activity_frac': high_activity_frac,
        }
        
        animal_features[animal_number] = features

    # Convert to DataFrame
    features_df = pd.DataFrame.from_dict(animal_features, orient='index')

    if output_path is not None:
        print(f"Saving features on {output_path}")
        features_df.to_csv(output_path, index=False)

    return features_df


def generate_temporal_features(animals_protocols, output_path=None):

    temporal_features = {}

    for animal_number in animals_protocols.keys():

        data = animals_protocols[animal_number].data
        values = data['values']

        # ── Signal conditioning ───────────────────────────────────────────────
        # 1. Detrend: remove linear trend across the recording (slow drift)
        from scipy.signal import detrend as sp_detrend
        values_detrended = pd.Series(
            sp_detrend(values.values, type='linear'),
            index=values.index
        )
        # Clip to zero (detrending can introduce negatives)
        values_detrended = values_detrended.clip(lower=0)

        # 2. z-score per animal so pattern, not magnitude, drives features
        v_mean = values_detrended.mean()
        v_std  = values_detrended.std()
        values_norm = (values_detrended - v_mean) / (v_std + 1e-9)

        # Use detrended signal for rhythmic / temporal features,
        # but raw values for magnitude-based features (total activity, etc.)
        # ─────────────────────────────────────────────────────────────────────

        # Sampling interval (minutes) for downstream metrics
        freq_minutes = values.index.to_series().diff().dt.total_seconds().median() / 60.0
        freq_minutes = freq_minutes if pd.notnull(freq_minutes) and freq_minutes > 0 else 60.0
        
        # Day vs Night activity (assuming zt_0_time = 20, so night is 20:00 to 8:00)
        daytime_mask = (data.index.hour >= 8) & (data.index.hour < 20)
        nighttime_mask = ~daytime_mask
        
        day_activity = values[daytime_mask].sum()
        night_activity = values[nighttime_mask].sum()
        day_night_ratio = day_activity / night_activity if night_activity > 0 else 0
        day_mean_activity = float(values[daytime_mask].mean()) if daytime_mask.any() else 0.0
        night_mean_activity = float(values[nighttime_mask].mean()) if nighttime_mask.any() else 0.0
        day_cv_activity = float(values[daytime_mask].std() / (values[daytime_mask].mean() + 1e-9)) if daytime_mask.any() else 0.0
        night_cv_activity = float(values[nighttime_mask].std() / (values[nighttime_mask].mean() + 1e-9)) if nighttime_mask.any() else 0.0
        daynight_log_ratio = float(np.log1p(day_activity) - np.log1p(night_activity)) if night_activity > 0 else float(np.log1p(day_activity))
        
        # Activity onset and offset (first and last significant activity in 24h cycle)
        # Calculate mean activity per hour
        hourly_activity = data.groupby(data.index.hour)['values'].mean()

        # Entropy of hourly distribution — normalised signal removes magnitude bias
        hourly_counts = values_norm.groupby(values_norm.index.hour).sum()
        hc_pos = hourly_counts.clip(lower=0)
        hourly_probs = hc_pos / (hc_pos.sum() + 1e-9)
        hourly_entropy = -float((hourly_probs * np.log(hourly_probs + 1e-12)).sum())
        hourly_entropy_norm = float(hourly_entropy / (np.log(len(hourly_counts)) + 1e-12))
        peak_to_mean_ratio = float(hourly_counts.max() / (hourly_counts.mean() + 1e-9))

        # Spectral rhythm strength — use normalised signal for cleaner spectrum
        sampling_seconds = freq_minutes * 60.0 if freq_minutes > 0 else 3600.0
        vals_centered = values_norm - values_norm.mean()
        fft_vals = np.fft.rfft(vals_centered)
        freqs_hz = np.fft.rfftfreq(len(vals_centered), d=sampling_seconds) if len(vals_centered) else np.array([0.0])

        def _power_at_period(hours):
            target_hz = 1.0 / (hours * 3600.0)
            if freqs_hz.size == 0:
                return 0.0
            idx = int(np.argmin(np.abs(freqs_hz - target_hz)))
            return float(np.abs(fft_vals[idx]) ** 2)

        power_24h = _power_at_period(24.0)
        power_12h = _power_at_period(12.0)
        power_8h = _power_at_period(8.0)
        harmonic_power = power_12h + power_8h
        rhythm_ratio_24_over_harm = float(power_24h / (harmonic_power + 1e-9))

        # Autocorrelation at 24h and 12h lags — normalised signal
        lag_24h = int(round((24 * 60) / freq_minutes))
        lag_12h = int(round((12 * 60) / freq_minutes))
        autocorr_24h = float(values_norm.autocorr(lag_24h)) if lag_24h >= 1 and lag_24h < len(values_norm) else 0.0
        autocorr_12h = float(values_norm.autocorr(lag_12h)) if lag_12h >= 1 and lag_12h < len(values_norm) else 0.0
        
        # Peak activity hour — normalised signal
        hourly_activity = values_norm.groupby(values_norm.index.hour).mean()
        peak_hour = hourly_activity.idxmax()
        
        # Activity fragmentation — bout detection on normalised signal
        # Use 75th percentile so ~25% of time is "active", creating meaningful gaps
        threshold = float(values_norm.quantile(0.75))
        above_threshold = values_norm > threshold

        def _run_lengths(mask: pd.Series):
            groups = mask.ne(mask.shift()).cumsum()
            lengths = mask.groupby(groups).size()
            states = mask.groupby(groups).first()
            bout_lengths = lengths[states].to_numpy()
            gap_lengths = lengths[~states].to_numpy()
            return bout_lengths, gap_lengths

        bout_lengths, gap_lengths = _run_lengths(above_threshold)
        # Convert run-lengths from sample counts to hours
        bout_lengths = bout_lengths * (freq_minutes / 60.0)
        gap_lengths  = gap_lengths  * (freq_minutes / 60.0)
        bouts = len(bout_lengths)
        transitions = int((above_threshold.astype(int).diff().abs() == 1).sum())
        hours_total = len(values) * (freq_minutes / 60.0)
        transitions_per_hour = float(transitions / (hours_total + 1e-9))
        mean_bout_len = float(np.mean(bout_lengths)) if len(bout_lengths) else 0.0
        median_bout_len = float(np.median(bout_lengths)) if len(bout_lengths) else 0.0
        p95_bout_len = float(np.percentile(bout_lengths, 95)) if len(bout_lengths) else 0.0
        bout_len_skew = float(stats.skew(bout_lengths)) if len(bout_lengths) >= 3 else 0.0
        bout_len_kurt = float(stats.kurtosis(bout_lengths, fisher=True, bias=False)) if len(bout_lengths) >= 4 else 0.0
        short_bout_frac = float(np.mean(bout_lengths <= 1.0)) if len(bout_lengths) else 0.0
        ibi_mean = float(np.mean(gap_lengths)) if len(gap_lengths) else 0.0
        ibi_median = float(np.median(gap_lengths)) if len(gap_lengths) else 0.0
        #ibi_min = float(np.min(gap_lengths)) if len(gap_lengths) else 0.0
        ibi_max = float(np.max(gap_lengths)) if len(gap_lengths) else 0.0
        #ibi_p25 = float(np.percentile(gap_lengths, 25)) if len(gap_lengths) else 0.0
        ibi_p75 = float(np.percentile(gap_lengths, 75)) if len(gap_lengths) else 0.0
        #ibi_iqr = ibi_p75 - ibi_p25
        ibi_p90 = float(np.percentile(gap_lengths, 90)) if len(gap_lengths) else 0.0
        ibi_skew = float(stats.skew(gap_lengths)) if len(gap_lengths) >= 3 else 0.0
        ibi_kurt = float(stats.kurtosis(gap_lengths, fisher=True, bias=False)) if len(gap_lengths) >= 4 else 0.0
        short_gap_frac = float(np.mean(gap_lengths <= 1.0)) if len(gap_lengths) else 0.0
        bout_len_cv = float(np.std(bout_lengths) / (np.mean(bout_lengths) + 1e-9)) if len(bout_lengths) else 0.0
        ibi_cv = float(np.std(gap_lengths) / (np.mean(gap_lengths) + 1e-9)) if len(gap_lengths) else 0.0

        # Day/night bout counts and rates (per 12h block)
        day_bouts = int(((above_threshold & daytime_mask).astype(int).diff() == 1).sum())
        night_bouts = int(((above_threshold & nighttime_mask).astype(int).diff() == 1).sum())
        bout_rate_day = day_bouts / 12.0
        bout_rate_night = night_bouts / 12.0
        bout_rate_ratio = bout_rate_day / (bout_rate_night + 1e-9)

        total_activity = values.sum()
        activity_per_bout = total_activity / (bouts + 1e-9)

        # Daily stability
        daily_totals = values.groupby(values.index.date).sum()
        daily_total_cv = float(daily_totals.std() / (daily_totals.mean() + 1e-9)) if len(daily_totals) > 1 else 0.0
        daily_peak_hours = []
        for d, s in values.groupby(values.index.date):
            hh_sum = s.groupby(s.index.hour).sum()
            daily_peak_hours.append(hh_sum.idxmax())
        daily_peak_std = float(np.std(daily_peak_hours)) if len(daily_peak_hours) > 1 else 0.0

        # Burstiness in fixed windows — normalised signal
        counts_30m = values_norm.resample('30T').sum()
        fano_30m = float(counts_30m.var() / (counts_30m.mean() + 1e-9)) if len(counts_30m) > 1 else 0.0

        # Cosinor: fit A*cos(2pi*t/24 + phi) + M via least squares — normalised signal
        t_hours = (values_norm.index.hour + values_norm.index.minute / 60.0).to_numpy()
        cos_t = np.cos(2 * np.pi * t_hours / 24.0)
        sin_t = np.sin(2 * np.pi * t_hours / 24.0)
        X_cos = np.column_stack([np.ones(len(t_hours)), cos_t, sin_t])
        try:
            coef, _, _, _ = np.linalg.lstsq(X_cos, values_norm.values, rcond=None)
            cosinor_mesor = float(coef[0])
            cosinor_amplitude = float(np.sqrt(coef[1]**2 + coef[2]**2))
            cosinor_acrophase = float(np.arctan2(-coef[2], coef[1]) % (2 * np.pi))
        except Exception:
            cosinor_mesor, cosinor_amplitude, cosinor_acrophase = 0.0, 0.0, 0.0

        # Non-parametric rhythm metrics — normalised signal
        hourly_vals = values_norm.resample('1h').mean().dropna()
        if len(hourly_vals) >= 10:
            m10 = float(hourly_vals.rolling(10, min_periods=10).mean().max())
            l5 = float(hourly_vals.rolling(5, min_periods=5).mean().min())
            relative_amplitude = float((m10 - l5) / (abs(m10) + abs(l5) + 1e-9))
        else:
            m10, l5, relative_amplitude = 0.0, 0.0, 0.0

        # Interdaily Stability (IS)
        if len(hourly_vals) >= 24:
            hourly_grand = values_norm.groupby(values_norm.index.hour).mean()
            p = len(values_norm)
            is_num = p * float(np.sum(hourly_grand.values ** 2))
            is_den = 24 * float(np.sum((values_norm.values - values_norm.mean()) ** 2) + 1e-9)
            interdaily_stability = float(is_num / is_den)
        else:
            interdaily_stability = 0.0

        # Intradaily Variability (IV)
        if len(values_norm) > 1:
            diffs_sq = float(np.sum(np.diff(values_norm.values) ** 2))
            dev_sq = float(np.sum((values_norm.values - values_norm.mean()) ** 2) + 1e-9)
            intradaily_variability = float((len(values_norm) * diffs_sq) / ((len(values_norm) - 1) * dev_sq))
        else:
            intradaily_variability = 0.0

        # Activity onset / offset (raw signal for magnitude threshold)
        hourly_activity_raw = values.groupby(values.index.hour).mean()
        onset_thresh = float(hourly_activity_raw.max() * 0.10)
        onset_hours_idx = hourly_activity_raw[hourly_activity_raw >= onset_thresh].index
        #activity_onset = float(onset_hours_idx.min()) if len(onset_hours_idx) > 0 else 0.0
        activity_offset = float(onset_hours_idx.max()) if len(onset_hours_idx) > 0 else 0.0
        #active_window_len = float(activity_offset - activity_onset) if activity_offset >= activity_onset else 0.0

        longest_inactive_run = float(np.max(gap_lengths)) if len(gap_lengths) else 0.0
        longest_active_run   = float(np.max(bout_lengths)) if len(bout_lengths) else 0.0

        active_fraction_day   = float((above_threshold & daytime_mask).sum() / (daytime_mask.sum() + 1e-9))
        active_fraction_night = float((above_threshold & nighttime_mask).sum() / (nighttime_mask.sum() + 1e-9))

        samples_per_hour = max(1, int(round(60.0 / freq_minutes)))
        rolling_std_1h = float(values_norm.rolling(samples_per_hour, min_periods=1).std().mean())

        # ── Night-only bout/IBI features ─────────────────────────────────────
        # Dominant mice tend to have longer, more consolidated bouts at night
        night_vals = values_norm[nighttime_mask]
        if len(night_vals) > 0:
            night_threshold = float(night_vals.quantile(0.75)) if night_vals.nunique() > 1 else float(night_vals.mean())
            night_above = night_vals > night_threshold
            night_bout_lens, night_gap_lens = _run_lengths(night_above)
            night_bout_lens = night_bout_lens * (freq_minutes / 60.0)
            night_gap_lens  = night_gap_lens  * (freq_minutes / 60.0)
        else:
            night_bout_lens, night_gap_lens = np.array([]), np.array([])

        night_ibi_median   = float(np.median(night_gap_lens))  if len(night_gap_lens) else 0.0
        night_ibi_cv       = float(np.std(night_gap_lens) / (np.mean(night_gap_lens) + 1e-9)) if len(night_gap_lens) else 0.0
        night_bout_len_mean= float(np.mean(night_bout_lens))   if len(night_bout_lens) else 0.0
        night_bout_len_cv  = float(np.std(night_bout_lens) / (np.mean(night_bout_lens) + 1e-9)) if len(night_bout_lens) else 0.0
        night_transitions_per_hour = float(
            (night_above.astype(int).diff().abs() == 1).sum() / (len(night_vals) * (freq_minutes / 60.0) + 1e-9)
        ) if len(night_vals) > 0 else 0.0

        # Onset latency: samples until first active bout after lights-off (ZT0 = 20:00)
        # Find the first nighttime sample where activity crosses threshold
        night_idx = np.where(np.asarray(nighttime_mask))[0]
        if len(night_idx) > 0 and len(night_bout_lens) > 0:
            first_night_above = np.where(night_above.values)[0]
            onset_latency_h = float(first_night_above[0] * (freq_minutes / 60.0)) if len(first_night_above) else float(len(night_vals) * (freq_minutes / 60.0))
        else:
            onset_latency_h = 0.0

        # First-2h night fraction: fraction of total night activity in first 2h after lights-off
        n_samples_2h = int(round(2 * 60 / freq_minutes))
        if len(night_vals) >= n_samples_2h and night_vals.sum() > 0:
            first_2h_frac = float(night_vals.iloc[:n_samples_2h].sum() / (night_vals.sum() + 1e-9))
        else:
            first_2h_frac = 0.0

        # Night activity Gini (concentration): dominant animals may have more concentrated bursts
        night_sorted = np.sort(night_vals.values)
        n_n = len(night_sorted)
        if n_n > 0 and night_sorted.sum() > 0:
            night_gini = (2.0 * np.sum((np.arange(1, n_n + 1)) * night_sorted) /
                          (n_n * night_sorted.sum() + 1e-9) - (n_n + 1) / n_n)
        else:
            night_gini = 0.0

        # Night peak hour (hour within night with highest mean activity)
        night_hourly = night_vals.groupby(night_vals.index.hour).mean()
        night_peak_hour = float(night_hourly.idxmax()) if len(night_hourly) > 0 else 0.0

        # Night/day activity ratio per bout (intensity when active at night vs day)
        night_activity_per_bout = float(night_vals.sum() / (len(night_bout_lens) + 1e-9))
        day_activity_per_bout   = float(values[daytime_mask].sum() / (day_bouts + 1e-9))
        night_day_intensity_ratio = float(night_activity_per_bout / (day_activity_per_bout + 1e-9))

        temporal_features[animal_number] = {
            'animal': animal_number,
            'day_activity': day_activity,
            'night_activity': night_activity,
            'day_night_ratio': day_night_ratio,
            'day_mean_activity': day_mean_activity,
            'night_mean_activity': night_mean_activity,
            'day_cv_activity': day_cv_activity,
            'night_cv_activity': night_cv_activity,
            'daynight_log_ratio': daynight_log_ratio,
            'peak_hour': peak_hour,
            'num_bouts': bouts,
            'num_bouts_day': day_bouts,
            'num_bouts_night': night_bouts,
            'bout_rate_day': bout_rate_day,
            'bout_rate_night': bout_rate_night,
            'bout_rate_ratio': bout_rate_ratio,
            'mean_bout_len': mean_bout_len,
            'median_bout_len': median_bout_len,
            'p95_bout_len': p95_bout_len,
            'bout_len_skew': bout_len_skew,
            'bout_len_kurt': bout_len_kurt,
            'bout_len_cv': bout_len_cv,
            'short_bout_frac': short_bout_frac,
            'transitions_per_hour': transitions_per_hour,
            'ibi_mean': ibi_mean,
            'ibi_median': ibi_median,
            #'ibi_min': ibi_min,
            'ibi_max': ibi_max,
            #'ibi_p25': ibi_p25,
            'ibi_p75': ibi_p75,
            #'ibi_iqr': ibi_iqr,
            'ibi_p90': ibi_p90,
            'ibi_skew': ibi_skew,
            'ibi_kurt': ibi_kurt,
            'ibi_cv': ibi_cv,
            'short_gap_frac': short_gap_frac,
            'day_frac': day_activity / (day_activity + night_activity + 1e-9),
            'night_frac': night_activity / (day_activity + night_activity + 1e-9),
            'day_minus_night': day_activity - night_activity,
            'activity_per_bout': activity_per_bout,
            'hourly_entropy': hourly_entropy,
            'hourly_entropy_norm': hourly_entropy_norm,
            'peak_to_mean_ratio': peak_to_mean_ratio,
            'power_24h': power_24h,
            'power_12h': power_12h,
            'power_8h': power_8h,
            'rhythm_ratio_24_over_harm': rhythm_ratio_24_over_harm,
            'daily_total_cv': daily_total_cv,
            'daily_peak_std': daily_peak_std,
            'fano_30m': fano_30m,
            'cosinor_mesor': cosinor_mesor,
            'cosinor_amplitude': cosinor_amplitude,
            'cosinor_acrophase': cosinor_acrophase,
            'relative_amplitude': relative_amplitude,
            'interdaily_stability': interdaily_stability,
            'intradaily_variability': intradaily_variability,
            #'activity_onset': activity_onset,
            'activity_offset': activity_offset,
            #'active_window_len': active_window_len,
            'longest_inactive_run': longest_inactive_run,
            'longest_active_run': longest_active_run,
            'active_fraction_day': active_fraction_day,
            'active_fraction_night': active_fraction_night,
            'rolling_std_1h': rolling_std_1h,
            'autocorr_24h': autocorr_24h,
            'autocorr_12h': autocorr_12h,
            # ── Night-specific features ──────────────────────────────────────
            'night_ibi_median': night_ibi_median,
            'night_ibi_cv': night_ibi_cv,
            'night_bout_len_mean': night_bout_len_mean,
            'night_bout_len_cv': night_bout_len_cv,
            'night_transitions_per_hour': night_transitions_per_hour,
            'night_onset_latency_h': onset_latency_h,
            'night_first_2h_frac': first_2h_frac,
            'night_gini': night_gini,
            'night_peak_hour': night_peak_hour,
            'night_activity_per_bout': night_activity_per_bout,
            'night_day_intensity_ratio': night_day_intensity_ratio,
  
        }

        

    temporal_df = pd.DataFrame.from_dict(temporal_features, orient='index')
    # if temporal_df['actual_rank'].notna().any():
    #temporal_df = temporal_df.sort_values('actual_rank')

    if output_path is not None:
        print(f"Saving temporal features on {output_path}")
        temporal_df.to_csv(output_path, index=False)

    return temporal_df

def combine_all_features(features_df, temporal_df, output_path):

    
    # Combine all features
    all_features = features_df.copy()
    for col in temporal_df.columns:
        if col not in ['animal', 'actual_rank']:
            all_features[col] = temporal_df[col]

    if output_path is not None:
        print(f"Saving all features on {output_path}")
        all_features.to_csv(output_path, index=False)

    feature_cols = [col for col in all_features.columns if col not in ['animal', 'actual_rank']]
    return all_features, feature_cols


def calculate_correlations(all_features, feature_cols, y, output_path=None):
    
  
    # Calculate correlations with rank
    
    correlations = {}

    for col in feature_cols:
        corr, pval = stats.spearmanr(all_features[col], y)
        correlations[col] = {'correlation': corr, 'p_value': pval}

    corr_df = pd.DataFrame.from_dict(correlations, orient='index')
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)

    if output_path is not None:
        print(f"Saving correlation on {output_path}")
        corr_df.to_csv(output_path)



    return corr_df






def build_animal_protocols(
        animals_files, 
        zt_0_time=20, 
        labels_dict={'cycle_types': ['LD'], 'test_labels': ['1_control_dl'], 'cycle_days': [1]}, 
        apply_filtering=True):

 
    animals_protocols = {}
    reads = []


    for k, v in animals_files.items():
        print(f"Animal2 {k}")
        if k not in reads:
            protocol = read_data(v[0], zt_0_time=zt_0_time, labels_dict=labels_dict)

            for i in range(1, len(v)):
                try:
                    protocol.concat_protocols(read_data(v[i], zt_0_time=zt_0_time, labels_dict=labels_dict), method='sum')
                except Exception as e:
                    print(f"Error trying to concatenate {v} with {v[i]}: {e}")
            
            if apply_filtering:
                protocol.resample('15T', method='sum')
                protocol.apply_filter(type = 'savgol')
            animals_protocols[f"animal_{k}"] = protocol
            reads.append(k)
        else:
            print(f"Animal {k} already read, skipping.")



    animals_by_day = {}
    for k in animals_protocols.keys():
        animals_by_day[k] = split_animal_protocol_by_day(animals_protocols, k)

    return animals_protocols, animals_by_day


def get_sorted_animals_files(individual_files, animals=None):

    animals_files = {}

    for file in individual_files:
        # Extract animal number from filename
        # Assuming format like "data_unwrapped_animal_1.txt"
        animal_number = int(file.split("_")[-1].split(".")[0])  # Gets "1" from "animal_1"

        if animals is not None and animal_number not in animals:
            continue
        
        if animal_number not in animals_files:
            animals_files[animal_number] = []
        
        animals_files[animal_number].append(file)

    # Sort files for each animal
    for animal in animals_files:
        animals_files[animal].sort()

    # Sort the dictionary by animal number
    animals_files = dict(sorted(animals_files.items(), key=lambda x: int(x[0])))

    for k,v in animals_files.items():
        print(f"Animal {k}: {len(v)}")

    return animals_files