# QuantFlow — Regime-Based Market Microstructure Trading

A regime-aware high-frequency trading system that dynamically adapts to evolving market microstructures. Built with deep generative models (DRA-GAN) for synthetic feature augmentation, MLP-based directional forecasting, and advanced position sizing using entropy and trend intensity metrics.

## Architecture

```
QuantFlow/
├── asset_x_strategy.py        # Strategy for Asset X (VPIN-gated, multi-SuperTrend)
├── asset_y_strategy.py        # Strategy for Asset Y (MACD-ADX confirmed, breakout-based)
├── utils/
│   ├── __init__.py
│   ├── position_sizer.py      # Entropy-based adaptive position sizing
│   └── indicators.py          # SuperTrend, VPIN, Z-VPIN indicators
├── models/                    # Pre-trained model weights (not tracked in git)
│   ├── gan_generator.keras
│   ├── base_mlp_model.keras
│   ├── base_mlp_scaler.pkl
│   ├── mlp_model.keras
│   └── mlp_scaler.pkl
├── requirements.txt
└── README.md
```

## Key Features

### Signal Generation
- **DRA-GAN + MLP Pipeline**: Butterworth-filtered price series → Z-score / momentum features → GAN generator for synthetic augmentation → MLP classifier for directional probability estimation
- **Multi-timeframe SuperTrend**: Three concurrent SuperTrend indicators (short/medium/long period) plus a composite variant for robust trend confirmation
- **VPIN Toxicity Filtering** (Asset X): Volume-Synchronized Probability of Informed Trading with Z-VPIN normalisation to gate trades based on order flow toxicity
- **MACD-ADX Confirmation** (Asset Y): MACD histogram momentum consistency combined with ADX trend strength for mid-frequency entry signals

### Risk Management
- **Entropy-Based Position Sizing**: Rolling Shannon entropy of log-returns modulates position size inversely — higher market uncertainty → smaller positions
- **Trend Intensity Index (TII)**: Amplifies sizing when directional conviction is strong
- **Sigmoid Time Decay**: Automatically reduces exposure as the trading session approaches close (inflection at 85% of session)
- **Turtle-Style Pyramiding**: Adds to winning positions in 0.5×N increments (N = rolling ATR proxy) up to the position cap

### Additional Components
- **Kalman Filter Denoising** (Asset X): Dual Kalman filters with different Q/R parameters for aggressive and conservative price smoothing
- **Adaptive Linear Regression Mean-Reversion** (Asset X): Sliding-window linear regression with optimal window selection (5–200 bars) for counter-trend signals
- **Early-Session Breakout** (Asset X): Cumulative price range analysis in the first ~5000 ticks to capture initial directional moves

## Performance

| Asset   | Annualised Return | Max Drawdown |
|---------|:-----------------:|:------------:|
| Asset X | ~30%              | ~6%          |
| Asset Y | ~25%              | ~6%          |

## Setup

### Prerequisites
- Python 3.10+
- TensorFlow 2.x
- scikit-learn, pandas, numpy, scipy, joblib

### Installation

```bash
pip install -r requirements.txt
```

### Running

```bash
# Asset X strategy
python asset_x_strategy.py config.json

# Asset Y strategy
python asset_y_strategy.py config.json
```

> **Note**: You need to provide your own backtester module and config file. The strategies expose `on_broadcast(state, ts, backtester)` and `on_timer(ts)` callbacks that can be plugged into any event-driven backtesting framework.

### Model Files

Place pre-trained model weights in the `models/` directory:
- `gan_generator.keras` — GAN generator for synthetic feature augmentation
- `base_mlp_model.keras` / `mlp_model.keras` — MLP directional classifiers
- `base_mlp_scaler.pkl` / `mlp_scaler.pkl` — Feature scalers

## How It Works

1. **Price Denoising**: Raw tick prices are passed through a 2nd-order Butterworth low-pass filter (cutoff = 0.15 Hz) to extract the underlying trend
2. **Feature Engineering**: Z-scores, momentum (1-bar, 5-bar), RSI, Bollinger Width, and Kaufman Efficiency Ratio are computed on the filtered series
3. **GAN Augmentation**: The feature sequence is fed into a trained GAN generator to produce synthetic distributional statistics (mean, std)
4. **MLP Prediction**: Combined real + synthetic features are scaled and classified by the MLP to produce a directional probability in [0, 1]
5. **Regime Detection**: Multi-timeframe SuperTrend, ADX, and trend persistence filters identify the current market regime
6. **Signal Gating**: VPIN (Asset X) or MACD-ADX (Asset Y) filters decide whether the regime is favourable for trading
7. **Position Sizing**: Entropy, TII, and time-decay factors are multiplied to compute the optimal position size
8. **Execution**: Orders are placed with Turtle-style pyramiding on favourable moves and hard stop-losses / take-profits

## License

MIT
