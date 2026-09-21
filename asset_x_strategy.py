"""
QuantFlow — Asset X Strategy

Regime-aware high-frequency trading strategy for Asset X.
Combines DRA-GAN + MLP directional forecasting, VPIN-based toxicity filtering,
multi-timeframe SuperTrend trend detection, entropy-based position sizing
with Turtle-style pyramiding, and adaptive linear-regression mean-reversion.
"""

import sys
import random
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter
from sklearn.linear_model import LinearRegression
from tensorflow import keras
import joblib

from utils.position_sizer import PositionSizer
from utils.indicators import (
    calc_supertrend,
    calc_supertrend_long,
    calc_supertrend_short,
    calc_supertrend_composite,
    vpin_fractional_ema,
    get_zvpin,
)

warnings.filterwarnings("ignore")

# ======================================================================
# Configuration
# ======================================================================

MA_WINDOW = 60
LOOK_BACK = 2
BAR_SIZE = "30s"
CLAMP_Z = 5.0
MODEL_EPS = 1e-9
TICKER = "AssetX"

# Model paths (relative)
GAN_PATH = "models/gan_generator.keras"
MLP_PATH = "models/base_mlp_model.keras"
SCALER_PATH = "models/base_mlp_scaler.pkl"

# Load pre-trained models
gen = keras.models.load_model(GAN_PATH, compile=False, safe_mode=False)
model = keras.models.load_model(MLP_PATH, compile=False, safe_mode=False)
scaler = joblib.load(SCALER_PATH)

# ======================================================================
# Strategy State (global, reset each session)
# ======================================================================

period = 14
alpha = 1 / period

tr_smooth = None
plus_dm_smooth = None
minus_dm_smooth = None
adx_smooth = None
VPIN_state = None
VPIN_value = None
in_strat = None
bucket_close = None
force_off = False
started = False
current_prob_filt = np.nan

x_hist = []
y_hist = []
sampled_data = []
gan_buffer = []
bucket_prices = []
bucket_volumes = []
buckets = []
VPIN_values = []
zVPIN_values = []
b = []
a = []
W, n, V, sigma_window = [300, 2, 100, 600]

trade_count = 0
entry_price = 0
pos = 0
pos_5k = 0
pos_ml = 0
N = 0
trade_in = 0
real_pos = 0

windowN = 20
pos_multi = 10

short_time = 8
long_time = 12
mid_time = 5
supertrend_period = 230
supertrend_atr = 4
supertrend_long_period = 1620
supertrend_long_atr = 2.77
MIN_HOLDING_PERIOD = 5
MIN_COOLDOWN_PERIOD = 750
NO_TRADE_PERIOD = 750
TAKE_PROFIT_AMOUNT = 6.5
sla = 1

ticks_in_trade = 0
ticks_since_last_signal = MIN_COOLDOWN_PERIOD + 1
ticks_in_trade2 = 0
ticks_since_last_signal2 = MIN_COOLDOWN_PERIOD + 1

curr_pos = 0
time = 0
holding_period = 3000
counter = 0

long_entry, long_exit, short_entry, short_exit = [0.66115865, 0.32670365, 0.364463, 0.6000999]

# Kalman filter parameters
P0 = 1.2
Q0 = 1e-8
R0 = 5e-5

P1 = 1.2
Q1 = 1e-7
R1 = 5e-4

pos_adv = 0
min_win_adv = 5
max_win_adv = 200
std_mult_adv = 30
signal_adv = 0

ini_vol = 300
late_exit = 500
tp = 1.3
sl = 0.8
vol = 0
start = 0

cum_max_price = -1
cum_min_price = 10000

entry_tick = 0
entryflag = 0
exitflag = 0
flag5k_vol = 0
posml = 0
flag_strat1 = 0
flag_strat2 = 0
flag_strat3 = 0
flag2hard = 0
flag2 = 0
pos2 = 0

adx_period = 14
cutoff = 0.15
fs = 1.0
order = 2
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype="low", analog=False)

HIGH = "PB10_T1"
LOW = "PB11_T1"
OPEN = "PB9_T1"
CLOSE = "Price"

defaults = {
    "V5_clipped": 0, "prob": np.nan, "prob_filt": np.nan, "conf": np.nan,
    "Spread": 0, "Tick": 0, "Rolling_Range": 0, "std600": 0, "TR": 0,
    "+DM": 0, "-DM": 0, "+DI": 0, "-DI": 0, "ADX": 0, "DX": 0, "TRadx": 0,
    "ATRadx": 0, "Smoothed_+DM": 0, "Smoothed_-DM": 0, "std_sigma_window": np.nan,
    "tr_smooth": 0, "plus_dm_smooth": 0, "minus_dm_smooth": 0,
    "In_Uptrend_T6": True, "In_Uptrend_T10": True, "In_Uptrend_T3": True,
    "IU": True, "in_uptrend": True, "in_downtrend": False, "uptrend": False,
    "downtrend": False, "High_mask": False, "Low_mask": False, "Mid_mask": False,
    "N": 0, "trN": 0, "Signal_DRA": 0, "Signal_adv": 0, "signal1": 0,
    "signal_5k": 0, "signal2": 0, "position1": 0, "Trend": 0, "Trend_Final": 0,
    "signal": 0, "Signal_MLP": 0, "Denoised_close_1": 0, "Denoised_close": 0,
    "position_size": 0, "ATRnew": np.nan, "Upper_adv": np.nan, "Lower_adv": np.nan,
    "Best_Window_adv": np.nan, "LR_adv": np.nan,
}

cols = [
    "Open", "High", "Low", "Close", "Volume", "V5_clipped", "prob", "prob_filt", "conf",
    "Spread", "Tick", "Rolling_Range", "std600", "TR", "+DM", "-DM", "+DI", "-DI", "ADX",
    "DX", "TRadx", "ATRadx", "Smoothed_+DM", "Smoothed_-DM", "std_sigma_window",
    "tr_smooth", "plus_dm_smooth", "minus_dm_smooth",
    "In_Uptrend_T6", "In_Uptrend_T10", "In_Uptrend_T3", "IU",
    "in_uptrend", "in_downtrend", "uptrend", "downtrend",
    "High_mask", "Low_mask", "Mid_mask", "N", "trN", "Signal_DRA", "Signal_adv",
    "signal1", "signal_5k", "signal2", "position1", "Trend", "Trend_Final", "signal",
    "Signal_MLP", "Denoised_close_1", "Denoised_close", "position_size", "ATRnew",
    "Upper_adv", "Lower_adv", "Best_Window_adv", "LR_adv", "TR_N", "Volatility",
    "Upperband8", "Lowerband8", "H-L8", "H-PC8", "L-PC8", "TR8", "ATR8",
    "Upperband12", "Lowerband12", "H-L12", "H-PC12", "L-PC12", "TR12", "ATR12",
    "Upperband5", "Lowerband5", "H-L5", "H-PC5", "L-PC5", "TR5", "ATR5",
    "Upperband_super", "Lowerband_super", "H-Lnew", "H-PCnew", "L-PCnew", "TRnew",
    "BestWindow_adv", "position_delta",
    "Price", "Time",
    "PB9_T1", "PB10_T1", "PB11_T1", "PB10_T3", "PB11_T3",
    "BB11_T3", "BB12_T5", "BB11_T4", "BB12_T4", "BB7_T4", "BB20", "BB8_T4", "BB19",
    "BB11_T8", "BB12_T8", "BB11_T12", "BB12_T12", "BB11_T5",
    "BB4_T5", "BB4_T8", "BB4_T10", "BB21", "BB22",
    "V5", "prob_fit",
    "entropy_raw", "factor_entropy", "factor_tii", "factor_decay",
    "raw_position_size",
    "BestWindow", "LR", "Upper", "Lower", "Signal", "Signal_ag",
    "vma", "Supertrend", "sup_signal", "In_Uptrend",
    "Units",
]

df = pd.DataFrame(columns=cols)
counter = 0
entry_price = 0
x1 = 0
x0 = 0


# ======================================================================
# Main broadcast callback
# ======================================================================

def on_broadcast(state, ts, backtester):
    """
    Called on every tick. Implements the full Asset X strategy pipeline:

    1. GAN + MLP directional probability estimation (every 30 ticks)
    2. Kalman-filtered price denoising
    3. Adaptive linear-regression mean-reversion signals
    4. Early-session breakout strategy (first ~5000 ticks)
    5. MLP regime-based signal generation (6000-21000 ticks)
    6. Multi-timeframe SuperTrend trend detection
    7. Conditional momentum & crossover entries
    8. VPIN toxicity filtering for trade gating
    9. Entropy-based position sizing with Turtle pyramiding
    """
    global period, order, alpha, counter
    global tr_smooth, plus_dm_smooth, minus_dm_smooth, adx_smooth
    global x_hist, y_hist, sampled_data, gan_buffer
    global current_prob_filt
    global bucket_prices, bucket_volumes, VPIN_state, VPIN_value
    global VPIN_values, zVPIN_values, b, a
    global started, windowN
    global pos, pos_5k, in_strat, pos_ml, N, force_off
    global trade_in, real_pos, pos_multi, W, n, V, sigma_window
    global short_time, long_time, buckets, mid_time
    global supertrend_period, supertrend_atr, supertrend_long_period, supertrend_long_atr
    global trade_count, entry_price, x1, x0
    global MIN_HOLDING_PERIOD, MIN_COOLDOWN_PERIOD, NO_TRADE_PERIOD
    global TAKE_PROFIT_AMOUNT, sla, ticks_in_trade, ticks_since_last_signal
    global ticks_in_trade2, ticks_since_last_signal2
    global curr_pos, time, holding_period
    global long_entry, long_exit, short_entry, short_exit
    global P0, Q0, R0, P1, Q1, R1
    global pos_adv, min_win_adv, max_win_adv, std_mult_adv, signal_adv
    global ini_vol, late_exit, tp, sl, vol, start, bucket_close
    global cum_max_price, cum_min_price
    global entry_tick, entryflag, exitflag
    global flag5k_vol, posml, adx_period
    global flag_strat1, flag_strat2, flag_strat3
    global flag2hard, flag2, pos2
    global df, curr_prob_val

    position = real_pos  # alias for clarity

    # --- End-of-day forced liquidation ---
    if counter >= 22339:
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
    df = df.apply(pd.to_numeric, errors="ignore")

    i = counter

    # ---- GAN + MLP probability estimation (every 30 ticks) ----
    if i % 30 == 0:
        _compute_gan_mlp_signal(i)

    df.at[i, "prob_filt"] = curr_prob_val

    # ---- Kalman filtering ----
    _kalman_update(i)

    # ---- Adaptive linear regression mean-reversion ----
    if i % 30 == 0 and i != 0:
        _adaptive_lr_signal(i)
    else:
        signal_adv = 0
    df.loc[i, "Signal_adv"] = signal_adv

    # ---- Early-session breakout (first ~5000 ticks) ----
    if i <= 5003:
        _early_session_breakout(i)

    # ---- MLP regime signal (6000-21000 ticks) ----
    if 6000 <= i <= 21000:
        if i % 30 == 0:
            _mlp_regime_signal(i)

    # ---- VPIN calculation & toxicity state ----
    _update_vpin(i)

    # ---- ADX calculation ----
    _update_adx(i)

    # ---- SuperTrend (multi-timeframe) ----
    if i != 0:
        df["In_Uptrend_T6"].iloc[i] = calc_supertrend(df, i, df["In_Uptrend_T6"].iloc[i - 1])
        df["In_Uptrend_T10"].iloc[i] = calc_supertrend_long(df, i, df["In_Uptrend_T10"].iloc[i - 1])
        df["In_Uptrend_T3"].iloc[i] = calc_supertrend_short(df, i, df["In_Uptrend_T3"].iloc[i - 1])
        df["IU"].iloc[i] = calc_supertrend_composite(df, i, df["IU"].iloc[i - 1])

    # ---- Trend state derivation ----
    _derive_trend_state(i)

    # ---- Conditional & momentum entries ----
    _compute_trend_signals(i)

    # ---- SMA crossover strategy ----
    _sma_crossover_strategy(i)

    df.loc[i, "signal1"] = curr_signal if flag_strat1 else 0
    df.loc[i, "Signal_DRA"] = df.loc[i, "signal1"] + df.loc[i, "signal2"]

    # ---- Order execution logic ----
    _execute_orders(i, backtester)

    # ---- Update position tracking ----
    if df.loc[df.index[i], "Low_mask"]:
        pos = pos + df.loc[df.index[i], "Signal_DRA"] + df.loc[df.index[i], "Signal_adv"]
    elif df.loc[df.index[i], "High_mask"]:
        pos = pos + df.loc[df.index[i], "Signal_DRA"]

    pos_5k = pos_5k + df.loc[df.index[i], "signal_5k"]

    if pos > 0:
        pos = 1
    elif pos < 0:
        pos = -1

    counter += 1


def _reset_session_state(state):
    """Reset all strategy state at the start of a new trading session."""
    global x1, x0, tr_smooth, plus_dm_smooth, minus_dm_smooth, adx_smooth
    global VPIN_state, VPIN_value, current_prob_filt, in_strat, bucket_close
    global x_hist, y_hist, sampled_data, gan_buffer
    global bucket_prices, bucket_volumes, VPIN_values, zVPIN_values, buckets, b, a
    global started, pos, pos_5k, pos_ml, N, force_off, trade_in, real_pos
    global pos_multi, W, n, V, sigma_window
    global short_time, long_time, mid_time
    global supertrend_period, supertrend_atr, supertrend_long_period, supertrend_long_atr
    global trade_count, entry_price, MIN_HOLDING_PERIOD, MIN_COOLDOWN_PERIOD
    global NO_TRADE_PERIOD, TAKE_PROFIT_AMOUNT, sla
    global ticks_in_trade, ticks_since_last_signal
    global ticks_in_trade2, ticks_since_last_signal2
    global curr_pos, time, holding_period
    global P0, Q0, R0, P1, Q1, R1
    global pos_adv, min_win_adv, max_win_adv, std_mult_adv, signal_adv
    global ini_vol, late_exit, tp, sl, vol, start
    global cum_max_price, cum_min_price
    global entry_tick, entryflag, exitflag, flag5k_vol, posml
    global flag_strat1, flag_strat2, flag_strat3, flag2hard, flag2, pos2
    global counter, adx_period, order, df

    x1 = state[TICKER]["Price"]
    x0 = state[TICKER]["Price"]

    tr_smooth = plus_dm_smooth = minus_dm_smooth = adx_smooth = None
    VPIN_state = VPIN_value = None
    current_prob_filt = np.nan
    in_strat = bucket_close = None

    x_hist, y_hist, sampled_data, gan_buffer = [], [], [], []
    bucket_prices, bucket_volumes = [], []
    VPIN_values, zVPIN_values, buckets = [], [], []
    started = False

    pos = pos_5k = pos_ml = N = trade_in = real_pos = 0
    force_off = False
    pos_multi = 10
    W, n, V, sigma_window = 300, 2, 100, 600
    short_time, long_time, mid_time = 8, 12, 5

    supertrend_period, supertrend_atr = 230, 4
    supertrend_long_period, supertrend_long_atr = 1620, 2.77

    trade_count = entry_price = 0
    MIN_HOLDING_PERIOD = 5
    MIN_COOLDOWN_PERIOD = 750
    NO_TRADE_PERIOD = 750
    TAKE_PROFIT_AMOUNT = 6.5
    sla = 1

    ticks_in_trade = ticks_since_last_signal = 0
    ticks_since_last_signal = MIN_COOLDOWN_PERIOD + 1
    ticks_in_trade2 = ticks_since_last_signal2 = 0
    ticks_since_last_signal2 = MIN_COOLDOWN_PERIOD + 1

    curr_pos = time = 0
    holding_period = 3000

    P0, Q0, R0 = 1.2, 1e-8, 5e-5
    P1, Q1, R1 = 1.2, 1e-7, 5e-4

    pos_adv = signal_adv = 0
    min_win_adv, max_win_adv, std_mult_adv = 5, 200, 30

    ini_vol, late_exit = 300, 500
    tp, sl = 1.3, 0.8
    vol = start = 0
    cum_max_price, cum_min_price = -1, 10000

    entry_price = entry_tick = pos = entryflag = exitflag = 0
    flag5k_vol = posml = 0
    flag_strat1 = flag_strat2 = flag_strat3 = 0
    flag2hard = flag2 = pos2 = 0
    counter = adx_period = 0
    adx_period = 14

    cutoff, fs, order = 0.15, 1.0, 2
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    first_row = defaults.copy()
    for col in state[TICKER]:
        val = state[TICKER][col]
        if hasattr(val, "iloc"):
            first_row[col] = val
        else:
            first_row[col] = val
    df.loc[0] = first_row


# ======================================================================
# Timer callback (unused, reserved for future)
# ======================================================================

def on_timer(ts):
    pass


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python asset_x_strategy.py <config.json>")
        sys.exit(1)

    config_file = sys.argv[1]

    # Import your backtester here
    # from backtester import Backtester, Side
    # bt = Backtester(config_file)
    # bt.run(broadcast_callback=lambda s, t: on_broadcast(s, t, bt),
    #        timer_callback=on_timer)

    print(f"[AssetX] Started at {datetime.now().strftime('%H:%M:%S')}")
    # bt.run(...)
    print(f"[AssetX] Finished at {datetime.now().strftime('%H:%M:%S')}")
