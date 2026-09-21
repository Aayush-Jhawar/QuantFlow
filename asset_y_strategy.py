"""
QuantFlow — Asset Y Strategy

Regime-aware high-frequency trading strategy for Asset Y.
Combines DRA-GAN + MLP directional forecasting, MACD-ADX confirmation,
breakout detection, and entropy-based position sizing with Turtle pyramiding.
"""

import sys
import random
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter
from tensorflow import keras
import joblib

from utils.position_sizer import PositionSizer

warnings.filterwarnings("ignore")

# ======================================================================
# Configuration
# ======================================================================

MA_WINDOW = 60
LOOK_BACK = 2
BAR_SIZE = "30s"
CLAMP_Z = 5.0
MODEL_EPS = 1e-9
TICKER = "AssetY"

HIGH = "PB10_T1"
LOW = "PB11_T1"
OPEN = "PB9_T1"
CLOSE = "Price"

# Model paths (relative)
GAN_PATH = "models/gan_generator.keras"
MLP_PATH = "models/mlp_model.keras"
SCALER_PATH = "models/mlp_scaler.pkl"

# Load pre-trained models
gen = keras.models.load_model(GAN_PATH, compile=False, safe_mode=False)
model = keras.models.load_model(MLP_PATH, compile=False, safe_mode=False)
scaler = joblib.load(SCALER_PATH)

# ======================================================================
# ML thresholds
# ======================================================================

vars_ml = [0.5, 0.5, 0.45, 0.4, 0.153]
long_entry_ml, long_exit, short_entry_ml, short_exit, conf_thresh = vars_ml

# Butterworth filter for price denoising
cutoff = 0.15
fs = 1.0
order = 2
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype="low", analog=False)

# ======================================================================
# Strategy state (reset each session)
# ======================================================================

x_hist = []
y_hist = []
sampled_data = []
gan_buffer = []

current_prob_filt = np.nan
conf_buffer = []
alpha_prob = 2 / (10 + 1)

look, body_frac_min, tp_breakout, sl_breakout = 232, 0.0025, 2, 0.5
look = int(look)

pos_ml = 0
active_flag_breakout = 0
long_flag_breakout = 0
short_flag_breakout = 0
entry_price_breakout = 0
entry_price = 0
position = 0
force_off = False
in_strat = None
trade_in = 0
pos_mid = 0
pos_breakout = 0
flag = 1
tp_price = 0
trailing_stop = 0
best_high = 0
best_low = 0
long_entry_breakout = 0
short_entry_breakout = 0
atr = 0
smoothed_plus_dm = 0
smoothed_minus_dm = 0
curr_prob_val = np.nan
curr_conf_val = np.nan
counter = 0
tp = 0.06
sl = 0.03
adx_period = 60
adx_pos = 0.6
adx_neg = 0.6
pos_multi = 5
fast_window = 10
slow_window = 30
sig_window = 60
window_N = 20
alpha_fast = 2 / (fast_window + 1)
alpha_slow = 2 / (slow_window + 1)
alpha_sig = 2 / (sig_window + 1)
alpha_atr = 1 / adx_period
fast_ema = 0
slow_ema = 0
macd_line = 0
macd_signal = 0
macd_hist = 0
N = 0

df = pd.DataFrame(
    columns=[
        "Time", "PB10_T7", "PB11_T7", "PB10_T1", "PB11_T1", "PB9_T1", "Price",
        "prob", "prob_filt", "conf", "signal_ml", "signal_breakout", "signal",
        "MACD_Fast_EMA", "MACD_Slow_EMA", "MACD_Line", "MACD_Signal",
        "MACD_Hist", "MACD_Hist_diff", "MACD_Hist_diff_pos", "MACD_Hist_diff_neg",
        "macd_pos_consistent", "macd_neg_consistent",
        "TR", "+DM", "-DM", "ATR", "Smoothed_+DM", "Smoothed_-DM",
        "+DI", "-DI", "DX", "ADX", "ADX_diff", "ADX_diff_pos", "ADX_diff_neg",
        "TR_N", "N", "signal_kb", "signal_ks",
        "adx_pos_consistent_base", "adx_neg_consistent_base",
        "adx_pos_percent_condition", "adx_neg_percent_condition",
    ]
)


# ======================================================================
# Main broadcast callback
# ======================================================================

def on_broadcast(state, ts, backtester):
    """
    Called on every tick. Implements the full Asset Y strategy pipeline:

    1. GAN + MLP probability estimation with EMA smoothing
    2. ML-driven regime entry/exit (confidence-gated)
    3. Breakout detection using lookback range analysis
    4. MACD-ADX confirmation for mid-frequency entries
    5. Entropy-based position sizing with Turtle pyramiding
    6. End-of-day forced liquidation
    """
    global pos_ml
    global active_flag_breakout, long_flag_breakout, short_flag_breakout
    global entry_price_breakout, entry_price, position
    global force_off, in_strat, trade_in, pos_mid, pos_breakout, flag
    global tp_price, trailing_stop, N, best_high, best_low
    global long_entry_breakout, short_entry_breakout
    global atr, smoothed_plus_dm, smoothed_minus_dm
    global curr_prob_val, curr_conf_val, counter, df
    global current_prob_filt, x_hist, y_hist, sampled_data, gan_buffer
    global fast_ema, slow_ema, macd_line, macd_signal, macd_hist

    # --- End-of-day forced liquidation ---
    if ts >= "06:13:25":
        df.loc[df.index[counter], "signal"] = -position
        if position > 0:
            backtester.place_sell(ticker=TICKER, qty=abs(position))
        elif position < 0:
            backtester.place_buy(ticker=TICKER, qty=abs(position))
        position = 0
        counter += 1
        return

    # --- Session open: reset all state ---
    if ts == "00:00:00":
        _reset_session_state(state)
    else:
        df.loc[len(df)] = state[TICKER]

    # ---- GAN + MLP probability estimation (every 30 ticks) ----
    if counter % 30 == 0:
        _compute_gan_mlp_signal()

    df.at[counter, "prob_filt"] = curr_prob_val
    df.at[counter, "conf"] = curr_conf_val

    prob = curr_prob_val
    confidence = curr_conf_val

    if counter < 6000:
        confidence = np.nan

    # ---- ML signal generation ----
    if np.isnan(confidence):
        df["signal_ml"][counter] = pos_ml
        flag = 0
    else:
        flag = 1

    if flag == 1:
        _ml_regime_signal(prob, confidence)

    # ---- Breakout detection ----
    _breakout_detection()

    # ---- Breakout position management ----
    _manage_breakout_position()

    # ---- ADX + MACD calculation ----
    _update_n_value()
    _update_macd()
    _update_adx_indicators()
    _compute_macd_adx_signals()

    # ---- Order execution ----
    _execute_orders(backtester)

    if pos_mid < 0:
        pos_mid = -1
    elif pos_mid > 0:
        pos_mid = 1

    counter += 1


def _reset_session_state(state):
    """Reset all strategy state at the start of a new trading session."""
    global pos_ml, active_flag_breakout, long_flag_breakout, short_flag_breakout
    global entry_price_breakout, entry_price, position, force_off, in_strat
    global trade_in, pos_mid, pos_breakout, flag, tp_price, trailing_stop
    global best_high, best_low, long_entry_breakout, short_entry_breakout
    global atr, smoothed_plus_dm, smoothed_minus_dm
    global curr_prob_val, curr_conf_val, counter
    global x_hist, y_hist, sampled_data, gan_buffer, current_prob_filt
    global df, fast_ema, slow_ema, macd_line, macd_signal, macd_hist

    pos_ml = 0
    active_flag_breakout = long_flag_breakout = short_flag_breakout = 0
    entry_price_breakout = entry_price = position = 0
    force_off = False
    in_strat = None
    trade_in = pos_mid = pos_breakout = 0
    flag = 1
    tp_price = trailing_stop = best_high = best_low = 0
    long_entry_breakout = short_entry_breakout = 0
    atr = smoothed_plus_dm = smoothed_minus_dm = 0
    curr_prob_val = curr_conf_val = np.nan
    current_prob_filt = np.nan
    counter = 0

    x_hist, y_hist, sampled_data, gan_buffer = [], [], [], []
    df = pd.DataFrame(columns=df.columns)
    df.loc[len(df)] = state[TICKER]
    fast_ema = df["Price"][0]
    slow_ema = df["Price"][0]
    macd_line = macd_signal = macd_hist = 0


def _compute_gan_mlp_signal():
    """Run GAN + MLP pipeline to estimate directional probability."""
    global current_prob_filt, curr_prob_val, curr_conf_val

    current_x = df.at[counter, "Price"]
    row = {"price": current_x, "index": counter}

    if len(sampled_data) == 0:
        x_hist[:] = [current_x] * order
        y_hist[:] = [current_x] * order
        current_y = current_x
    else:
        val = b[0] * current_x
        for k in range(1, order + 1):
            val += b[k] * x_hist[k - 1]
        for k in range(1, order + 1):
            val -= a[k] * y_hist[k - 1]
        current_y = val
        x_hist.insert(0, current_x)
        x_hist.pop()
        y_hist.insert(0, current_y)
        y_hist.pop()

    row["filtered_price"] = current_y
    sampled_data.append(row)
    curr_idx = len(sampled_data) - 1

    rm_win = sampled_data[max(0, curr_idx - MA_WINDOW + 1) : curr_idx + 1]
    rm_vals = [x["filtered_price"] for x in rm_win]
    rm_val = np.mean(rm_vals)
    rs_val = np.std(rm_vals, ddof=1) if len(rm_vals) > 1 else 0.0

    z = max(-CLAMP_Z, min(CLAMP_Z, (current_y - rm_val) / (rs_val + MODEL_EPS)))

    m1_raw = (
        (current_y - sampled_data[curr_idx - 1]["filtered_price"]) / (rs_val + MODEL_EPS)
        if curr_idx > 0
        else 0.0
    )
    m1 = max(-CLAMP_Z, min(CLAMP_Z, m1_raw))

    m5_raw = (
        (current_y - sampled_data[curr_idx - 5]["filtered_price"]) / (rs_val + MODEL_EPS)
        if curr_idx > 4
        else 0.0
    )
    m5 = max(-CLAMP_Z, min(CLAMP_Z, m5_raw))

    gan_buffer.append([z, m1, m5])

    if len(gan_buffer) >= LOOK_BACK:
        seq_arr = np.array(gan_buffer[-LOOK_BACK:], dtype=np.float32)
        gan_input_vec = seq_arr.reshape(1, LOOK_BACK, 3)
        gan_out = gen.predict(gan_input_vec, verbose=0)
        g_mean = np.mean(gan_out)
        g_std = np.std(gan_out)
    else:
        g_mean = g_std = 0.0

    if curr_idx == 0:
        delta = 0.0
    else:
        delta = current_y - sampled_data[curr_idx - 1]["filtered_price"]

    u = delta if delta > 0 else 0.0
    d = abs(delta) if delta < 0 else 0.0
    sampled_data[curr_idx]["u"] = u
    sampled_data[curr_idx]["d"] = d

    rsi_win = sampled_data[max(0, curr_idx - 14 + 1) : curr_idx + 1]
    avg_gain = np.mean([x["u"] for x in rsi_win])
    avg_loss = np.mean([x["d"] for x in rsi_win])
    rs_calc = avg_gain / (avg_loss + MODEL_EPS)
    rsi_val = (100 - (100 / (1 + rs_calc))) / 100.0

    w_win = sampled_data[max(0, curr_idx - 20 + 1) : curr_idx + 1]
    w_vals = [x["filtered_price"] for x in w_win]
    w_ma = np.mean(w_vals)
    w_std = np.std(w_vals, ddof=1) if len(w_vals) > 1 else 0.0
    width_val = ((w_ma + 2 * w_std) - (w_ma - 2 * w_std)) / (w_ma + MODEL_EPS)

    ker_change = (
        abs(current_y - sampled_data[curr_idx - 20]["filtered_price"])
        if curr_idx >= 20
        else 0.0
    )
    ker_win = sampled_data[max(0, curr_idx - 20 + 1) : curr_idx + 1]
    ker_path = sum([x["u"] + x["d"] for x in ker_win])
    ker_val = ker_change / (ker_path + MODEL_EPS)

    X_tech = [rsi_val, width_val, ker_val]

    tiny_vals = [
        x["filtered_price"]
        for x in sampled_data[max(0, curr_idx - LOOK_BACK + 1) : curr_idx + 1]
    ]
    ma20_e = np.mean(tiny_vals)
    std20_e = np.std(tiny_vals, ddof=1) if len(tiny_vals) > 1 else 0.0
    boll_width_e = ((ma20_e + 2 * std20_e) - (ma20_e - 2 * std20_e)) / (ma20_e + MODEL_EPS)
    tr_val_e = abs(tiny_vals[-1] - tiny_vals[-2]) if len(tiny_vals) > 1 else 0.0
    slope_e = (tiny_vals[-1] - tiny_vals[0]) / (len(tiny_vals) - 1) if len(tiny_vals) > 1 else 0.0

    X_extra = [0.0, boll_width_e, 0.0, tr_val_e, 0.0, slope_e, 0.0, 0.0]

    feats = np.array([[g_mean, g_std] + X_tech + X_extra], dtype=np.float32)
    feats_scaled = scaler.transform(feats)
    prob_val = float(model.predict(feats_scaled, verbose=0).flatten()[0])

    df.at[counter, "prob"] = prob_val

    if np.isnan(current_prob_filt):
        current_prob_filt = prob_val
    else:
        current_prob_filt = alpha_prob * prob_val + (1 - alpha_prob) * current_prob_filt

    diff = abs(current_prob_filt - 0.5)
    conf_buffer.append(diff)
    if len(conf_buffer) > 10:
        conf_buffer.pop(0)

    current_conf = sum(conf_buffer) / 10 if len(conf_buffer) == 10 else np.nan

    curr_prob_val = current_prob_filt
    curr_conf_val = current_conf


def _ml_regime_signal(prob, confidence):
    """Generate ML-based regime entry/exit signals (confidence-gated)."""
    global pos_ml

    conf_ok = confidence > conf_thresh

    long_entry_cond = prob > long_entry_ml
    short_entry_cond = prob < short_entry_ml
    long_exit_cond = prob < long_exit
    short_exit_cond = prob > short_exit

    if pos_ml == 0 and conf_ok:
        if long_entry_cond and not short_entry_cond:
            pos_ml = 1
        elif short_entry_cond and not long_entry_cond:
            pos_ml = -1
    elif pos_ml == 1:
        if long_exit_cond and not short_entry_cond:
            pos_ml = 0
    elif pos_ml == -1:
        if short_exit_cond and not long_entry_cond:
            pos_ml = 0

    df["signal_ml"][counter] = pos_ml


def _breakout_detection():
    """Detect breakout entries using lookback range analysis."""
    global active_flag_breakout, long_entry_breakout, short_entry_breakout

    if counter >= look:
        active_flag_breakout = 1

    if active_flag_breakout == 1 and long_flag_breakout == 0 and short_flag_breakout == 0:
        prev_idx = max(0, counter - look)

        recent_range = (
            df[HIGH][max(0, counter - look + 1) : counter + 1]
            - df[LOW][max(0, counter - look + 1) : counter + 1]
        )
        avg_range = recent_range.mean() if len(recent_range) > 0 else 0.0
        last_body = abs(df[CLOSE][counter] - df[OPEN][counter])

        c_now = df[CLOSE][counter]
        c_prev = df[CLOSE][prev_idx]
        long_entry_breakout = (c_now > c_prev) and (last_body >= body_frac_min * (avg_range + 1e-12))
        short_entry_breakout = (c_now < c_prev) and (last_body >= body_frac_min * (avg_range + 1e-12))


def _manage_breakout_position():
    """Manage open breakout positions (trailing stop, take profit)."""
    global long_flag_breakout, short_flag_breakout
    global entry_price_breakout, tp_price, trailing_stop
    global best_high, best_low, long_entry_breakout, short_entry_breakout

    if long_entry_breakout:
        long_flag_breakout = 1
        entry_price_breakout = df[CLOSE][counter]
        tp_price = entry_price_breakout * (1.0 + tp_breakout)
        trailing_stop = entry_price_breakout * (1.0 - sl_breakout)
        best_high = entry_price_breakout
        df["signal_breakout"][counter] = 1
        long_entry_breakout = 0

    elif short_entry_breakout:
        short_flag_breakout = 1
        entry_price_breakout = df[CLOSE][counter]
        tp_price = entry_price_breakout * (1.0 - tp_breakout)
        trailing_stop = entry_price_breakout * (1.0 + sl_breakout)
        best_low = entry_price_breakout
        df["signal_breakout"][counter] = -1
        short_entry_breakout = 0

    if long_flag_breakout == 1:
        p = df[CLOSE][counter]
        if p > best_high:
            best_high = p
            trailing_stop = best_high * (1.0 - sl_breakout)
        if p >= tp_price:
            df["signal_breakout"][counter] = -1
            long_flag_breakout = 0
        if p <= trailing_stop:
            df["signal_breakout"][counter] = -1
            long_flag_breakout = 0

    if short_flag_breakout == 1:
        p = df[CLOSE][counter]
        if p < best_low:
            best_low = p
            trailing_stop = best_low * (1.0 + sl_breakout)
        if p <= tp_price:
            df["signal_breakout"][counter] = 1
            short_flag_breakout = 0
        if p >= trailing_stop:
            df["signal_breakout"][counter] = 1
            short_flag_breakout = 0


def _update_n_value():
    """Update the N (ATR proxy) value for Turtle pyramiding."""
    global N

    high_i = df.at[counter, "PB10_T1"]
    low_i = df.at[counter, "PB11_T1"]
    prev_close_i = df.at[counter - 1, "Price"] if counter > 0 else np.nan
    prev_close_i = float(prev_close_i)

    if high_i == "" or low_i == "":
        high_i = low_i = 0
    else:
        high_i = float(high_i)
        low_i = float(low_i)

    tr_i = np.nanmax([high_i - low_i, abs(high_i - prev_close_i), abs(prev_close_i - low_i)])
    df.at[counter, "TR_N"] = tr_i

    if counter >= window_N - 1:
        df.at[counter, "N"] = df["TR_N"].iloc[counter - window_N + 1 : counter + 1].mean()
    else:
        df.at[counter, "N"] = np.nan


def _update_macd():
    """Update MACD (Fast EMA, Slow EMA, Signal, Histogram)."""
    global fast_ema, slow_ema, macd_line, macd_signal, macd_hist

    if counter == 0:
        fast_ema = df["Price"][counter]
        slow_ema = df["Price"][counter]
        macd_line = macd_signal = macd_hist = 0
    else:
        fast_ema = alpha_fast * df["Price"][counter] + (1 - alpha_fast) * fast_ema
        slow_ema = alpha_slow * df["Price"][counter] + (1 - alpha_slow) * slow_ema
        macd_line = fast_ema - slow_ema
        macd_signal = alpha_sig * macd_line + (1 - alpha_sig) * macd_signal
        macd_hist = macd_line - macd_signal

    df.loc[counter, "MACD_Fast_EMA"] = fast_ema
    df.loc[counter, "MACD_Slow_EMA"] = slow_ema
    df.loc[counter, "MACD_Line"] = macd_line
    df.loc[counter, "MACD_Signal"] = macd_signal
    df.loc[counter, "MACD_Hist"] = macd_hist

    df["MACD_Hist_diff"][counter] = (
        df["MACD_Hist"][counter] - df["MACD_Hist"][counter - 1] if counter > 0 else 0
    )
    df["MACD_Hist_diff_pos"][counter] = int(df["MACD_Hist_diff"][counter] > 0)
    df["MACD_Hist_diff_neg"][counter] = int(df["MACD_Hist_diff"][counter] < 0)

    df["macd_pos_consistent"][counter] = (
        df["MACD_Hist_diff_pos"].iloc[max(0, counter - 4) : counter + 1].sum() == 5
    )
    df["macd_neg_consistent"][counter] = (
        df["MACD_Hist_diff_neg"].iloc[max(0, counter - 4) : counter + 1].sum() == 5
    )


def _update_adx_indicators():
    """Update ADX and directional movement indicators."""
    high_i_atr = df["PB10_T7"][counter]
    low_i_atr = df["PB11_T7"][counter]
    close_i_atr = df["PB9_T1"].shift(1)[counter]

    if high_i_atr == "" or low_i_atr == "":
        high_i_atr = low_i_atr = np.nan
    else:
        high_i_atr = float(high_i_atr)
        low_i_atr = float(low_i_atr)

    if close_i_atr is None or close_i_atr == "":
        close_i_atr = np.nan
    else:
        close_i_atr = float(close_i_atr)

    df["TR"][counter] = np.nanmax([
        high_i_atr - low_i_atr,
        abs(high_i_atr - close_i_atr),
        abs(low_i_atr - close_i_atr),
    ])

    move_up_high = df["PB10_T7"].shift(1)[counter]
    move_down_low = df["PB11_T7"].shift(1)[counter]

    if move_up_high == "" or move_down_low == "":
        move_up_high = move_down_low = np.nan
    elif move_up_high is None or move_down_low is None:
        move_up_high = move_down_low = np.nan
    else:
        move_up_high = float(move_up_high)
        move_down_low = float(move_down_low)

    move_up = float(df["PB10_T7"][counter]) - move_up_high if counter > 0 else 0
    move_down = move_down_low - float(df["PB11_T7"][counter]) if counter > 0 else 0
    plus_dm = move_up if (move_up > move_down) and (move_up > 0) else 0
    minus_dm = move_down if (move_down > move_up) and (move_down > 0) else 0
    df["+DM"][counter] = plus_dm
    df["-DM"][counter] = minus_dm

    df["ATR"][counter] = df["TR"].iloc[: counter + 1].ewm(alpha=1 / adx_period, adjust=False).mean().iloc[-1]
    df["N"][counter] = df["TR"].iloc[: counter + 1].ewm(alpha=1 / 20, adjust=False).mean().iloc[-1]
    df["Smoothed_+DM"][counter] = df["+DM"].iloc[: counter + 1].ewm(alpha=1 / adx_period, adjust=False).mean().iloc[-1]
    df["Smoothed_-DM"][counter] = df["-DM"].iloc[: counter + 1].ewm(alpha=1 / adx_period, adjust=False).mean().iloc[-1]
    df["+DI"][counter] = (df["Smoothed_+DM"][counter] / df["ATR"][counter]) * 100
    df["-DI"][counter] = (df["Smoothed_-DM"][counter] / df["ATR"][counter]) * 100
    dx = (abs(df["+DI"][counter] - df["-DI"][counter]) / (df["+DI"][counter] + df["-DI"][counter])) * 100
    dx = 0 if (np.isinf(dx) or np.isnan(dx)) else dx
    df["DX"][counter] = dx
    df["ADX"][counter] = df["DX"].iloc[: counter + 1].ewm(alpha=1 / adx_period, adjust=False).mean().iloc[-1]

    df["ADX_diff"][counter] = df["ADX"][counter] - df["ADX"][counter - 1] if counter > 0 else np.nan
    df["ADX_diff_pos"][counter] = int(df["ADX_diff"][counter] > 0)
    df["ADX_diff_neg"][counter] = int(df["ADX_diff"][counter] < 0)

    df["adx_pos_consistent_base"][counter] = (
        df["ADX_diff_pos"].iloc[max(0, counter - 4) : counter + 1].sum() == 5
    )
    df["adx_neg_consistent_base"][counter] = (
        df["ADX_diff_neg"].iloc[max(0, counter - 4) : counter + 1].sum() == 5
    )

    adx_at_t_minus_4 = df["ADX"][counter - 4] if counter - 4 >= 0 else np.nan
    adx_at_current_t = df["ADX"][counter]

    valid_indices = (not pd.isna(adx_at_t_minus_4)) and (adx_at_t_minus_4 != 0)
    if valid_indices:
        df["adx_pos_percent_condition"][counter] = (
            ((adx_at_current_t - adx_at_t_minus_4) / adx_at_t_minus_4) >= adx_pos
        )
        df["adx_neg_percent_condition"][counter] = (
            ((adx_at_current_t - adx_at_t_minus_4) / adx_at_t_minus_4) >= adx_neg
        )

    df.loc[counter, "adx_pos_consistent"] = (
        df["adx_pos_consistent_base"][counter] and df["adx_pos_percent_condition"][counter]
    )
    df.loc[counter, "adx_neg_consistent"] = (
        df["adx_pos_consistent_base"][counter] and df["adx_neg_percent_condition"][counter]
    )


def _compute_macd_adx_signals():
    """Compute MACD + ADX confirmation signals."""
    buy_signal = int(df["macd_pos_consistent"][counter] and df["adx_pos_consistent"][counter])
    sell_signal = int(df["macd_neg_consistent"][counter] and df["adx_neg_consistent"][counter])
    df["signal_kb"][counter] = buy_signal == 1
    df["signal_ks"][counter] = sell_signal == 1


def _execute_orders(backtester):
    """Execute order logic based on strategy signals."""
    global position, in_strat, trade_in, pos_breakout, entry_price, N, pos_mid, counter

    # --- ML signal execution ---
    if df["signal_ml"][counter] != 0 and in_strat != "ml":
        force_off_local = True
        in_strat = "ml"
        if position > 0:
            if df["signal_ml"][counter] == 1:
                df["signal"].iloc[counter] = 100 - position
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 100
            elif df["signal_ml"][counter] == -1:
                df["signal"].iloc[counter] = -position
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 0
                trade_in = 6
        elif position < 0:
            if df["signal_ml"][counter] == -1:
                df["signal"].iloc[counter] = -100 - position
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = -100
            elif df["signal_ml"][counter] == 1:
                df["signal"].iloc[counter] = -position
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 0
                trade_in = 6
        elif position == 0:
            if df["signal_ml"][counter] == 1:
                df["signal"].iloc[counter] = 100
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 100
            elif df["signal_ml"][counter] == -1:
                df["signal"].iloc[counter] = -100
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = -100
        counter += 1
        return

    # --- Continue ML regime ---
    if in_strat == "ml":
        if position == 0:
            if trade_in > 1:
                trade_in -= 1
                counter += 1
                return
            elif trade_in == 1:
                df["signal"][counter] = df["signal_ml"][counter - 6] * 100
                if df["signal"][counter] > 0:
                    backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                elif df["signal"][counter] < 0:
                    backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = df["signal_ml"][counter - 6] * 100
                trade_in = 0
            elif trade_in == 0:
                df["signal"][counter] = df["signal_ml"][counter] * 100
                if df["signal"][counter] > 0:
                    backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                elif df["signal"][counter] < 0:
                    backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = df["signal_ml"][counter] * 100
        elif position < 0:
            if df["signal_ml"][counter] == 1:
                df["signal"][counter] = -position
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 0
        elif position > 0:
            if df["signal_ml"][counter] == -1:
                df["signal"][counter] = -position
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 0

    # --- Breakout strategy execution ---
    if not force_off and in_strat != "mid":
        if counter == 6000:
            df["signal"][counter] = -position
            if df["signal"][counter] > 0:
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
            elif df["signal"][counter] < 0:
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
            in_strat = None
            pos_breakout = 0
            position = 0
            counter += 1
            return

        if df.loc[df.index[counter], "signal_breakout"] == 1:
            if position < 0:
                df["signal"].iloc[counter] = -position
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = pos_breakout = 0
                in_strat = None
                counter += 1
                return
            elif position == 0:
                sizer = PositionSizer()
                data = sizer.get_position_sizing_series(df[: counter + 1])
                df["signal"].iloc[counter] = min(100, pos_multi * data["position_size"].iloc[counter])
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = df["signal"].iloc[counter]
                in_strat = "breakout"
                N = df["N"].iloc[counter]
                pos_breakout = df["signal_breakout"].iloc[counter]
                entry_price = df["Price"].iloc[counter]
                counter += 1
                return

        elif df.loc[df.index[counter], "signal_breakout"] == -1:
            if position > 0 and pos_breakout > 0:
                df["signal"].iloc[counter] = -position
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = pos_breakout = 0
                in_strat = None
                counter += 1
                return
            elif position == 0:
                sizer = PositionSizer()
                data = sizer.get_position_sizing_series(df[: counter + 1])
                df["signal"].iloc[counter] = -min(100, pos_multi * data["position_size"].iloc[counter])
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = df["signal"].iloc[counter]
                in_strat = "breakout"
                N = df["N"].iloc[counter]
                pos_breakout = df["signal_breakout"].iloc[counter]
                entry_price = df["Price"].iloc[counter]
                counter += 1
                return

        # Turtle pyramiding
        if position > 0 and df["Price"][counter] > entry_price + N * 0.5:
            sizer = PositionSizer()
            data = sizer.get_position_sizing_series(df[: counter + 1])
            df["signal"][counter] = min(100 - position, pos_multi * data["position_size"].iloc[counter])
            backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
            position += df["signal"][counter]
            entry_price += N * 0.5
        elif position < 0 and df["Price"][counter] < entry_price - N * 0.5:
            sizer = PositionSizer()
            data = sizer.get_position_sizing_series(df[: counter + 1])
            df["signal"][counter] = max(-100 - position, -pos_multi * data["position_size"].iloc[counter])
            backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
            position += df["signal"][counter]
            entry_price -= N * 0.5

    # --- MACD-ADX mid-frequency strategy ---
    if not force_off and in_strat != "breakout":
        buy = df["signal_kb"].iloc[counter]
        sell = df["signal_ks"].iloc[counter]
        price = df["Price"].iloc[counter]

        if position == 0:
            if buy == 1:
                sizer = PositionSizer()
                data = sizer.get_position_sizing_series(df[: counter + 1])
                entry_price = price
                df.loc[df.index[counter], "signal"] = min(100, pos_multi * data["position_size"].iloc[counter])
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = df["signal"].iloc[counter]
                N = df["N"].iloc[counter]
                in_strat = "mid"
            elif sell == 1:
                sizer = PositionSizer()
                data = sizer.get_position_sizing_series(df[: counter + 1])
                entry_price = price
                df.loc[df.index[counter], "signal"] = -min(100, pos_multi * data["position_size"].iloc[counter])
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = df["signal"].iloc[counter]
                N = df["N"].iloc[counter]
                in_strat = "mid"

        elif position > 0:
            if price >= entry_price * (1 + tp):
                df.loc[df.index[counter], "signal"] = -position
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 0
                in_strat = None
            elif price <= entry_price * (1 - sl):
                df.loc[df.index[counter], "signal"] = -position
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 0
                in_strat = None
            elif sell == 1:
                df.loc[df.index[counter], "signal"] = -position
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 0
                in_strat = None
            elif price >= entry_price + 0.5 * N:
                sizer = PositionSizer()
                data = sizer.get_position_sizing_series(df[: counter + 1])
                df["signal"].iloc[counter] = min(100 - position, pos_multi * data["position_size"].iloc[counter])
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position += df["signal"].iloc[counter]
                entry_price += 0.5 * N

        elif position < 0:
            if price <= entry_price * (1 - tp):
                df.loc[df.index[counter], "signal"] = -position
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 0
            elif price >= entry_price * (1 + sl):
                df.loc[df.index[counter], "signal"] = -position
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 0
            elif buy == 1:
                df.loc[df.index[counter], "signal"] = -position
                backtester.place_buy(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position = 0
            elif price <= entry_price - 0.5 * N:
                sizer = PositionSizer()
                data = sizer.get_position_sizing_series(df[: counter + 1])
                df["signal"].iloc[counter] = max(-100 - position, -pos_multi * data["position_size"].iloc[counter])
                backtester.place_sell(ticker=TICKER, qty=abs(df["signal"].iloc[counter]))
                position += df["signal"].iloc[counter]
                entry_price -= 0.5 * N


# ======================================================================
# Timer callback
# ======================================================================

def on_timer(ts):
    pass


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python asset_y_strategy.py <config.json>")
        sys.exit(1)

    config_file = sys.argv[1]

    # Import your backtester here
    # from backtester import Backtester
    # bt = Backtester(config_file)

    print(f"[AssetY] Started at {datetime.now().strftime('%H:%M:%S')}")
    # bt.run(broadcast_callback=lambda s, t: on_broadcast(s, t, bt),
    #        timer_callback=on_timer)
    print(f"[AssetY] Finished at {datetime.now().strftime('%H:%M:%S')}")
