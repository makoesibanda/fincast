# Standard library imports
import os
import json
import math
import logging
from datetime import datetime, timedelta

# Data and numerical processing
import numpy as np
import pandas as pd
import joblib

# PyTorch for model loading and inference
import torch
import torch.nn as nn

# Live market data from Yahoo Finance
import yfinance as yf


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Run on GPU if available, otherwise CPU is fine for inference
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')


def build_classifier_head(input_size, dropout=0.2):
    # Builds the two-layer MLP used as the direction classifier in all DL models.
    # A stronger two-layer design was chosen over a single linear layer in v2
    # to give the model more capacity to learn up/down directional patterns.
    return nn.Sequential(
        nn.Linear(input_size, input_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(input_size, 1),
        nn.Sigmoid()
    )


class GRUModel(nn.Module):
    # GRU model trained on 30-day windows of 19 technical features.
    # Outputs a predicted next-day return and a direction probability.
    # Primary model in the HFM ensemble — carries 90% of the final prediction weight.
    def __init__(self, input_size=19, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        # Dropout between GRU layers only — not applied on a single layer
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout  = nn.Dropout(dropout)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = build_classifier_head(hidden_size, dropout)

    def forward(self, x):
        output, _ = self.gru(x)
        # Only the last time step is used — it holds the accumulated sequence context
        last_step = self.dropout(output[:, -1, :])
        return self.reg_head(last_step), self.cls_head(last_step)


class PositionalEncoding(nn.Module):
    # Injects position information into the Transformer input.
    # Transformers have no built-in sense of order, so sine and cosine signals
    # at different frequencies are added to make each time step distinguishable.
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe        = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term  = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[:d_model // 2])

        # Registered as a buffer so it moves to GPU but is not a trainable parameter
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    # Transformer encoder used as the secondary model in the HFM ensemble (10% weight).
    # Self-attention allows it to look across the full 30-day window simultaneously,
    # capturing longer-range dependencies that step-by-step GRUs can sometimes miss.
    def __init__(self, input_size=19, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.2):
        super().__init__()
        # Project 19 input features up to d_model dimensions before attention layers
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout  = nn.Dropout(dropout)
        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = build_classifier_head(d_model, dropout)

    def forward(self, x):
        x         = self.pos_enc(self.input_proj(x))
        x         = self.encoder(x)
        last_step = self.dropout(x[:, -1, :])
        return self.reg_head(last_step), self.cls_head(last_step)


class ModelLoader:
    # Singleton that keeps all models loaded in memory for the life of the server.
    # Loading PyTorch models from disk on every request would be too slow,
    # so we load once at startup and reuse the same objects.
    _instance = None

    def __new__(cls):
        # Return the existing instance if one already exists
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def init(self):
        # Skip if already initialised
        if self._ready:
            return

        # Load feature config saved at the end of notebook training.
        # This tells us which 19 features to use and in what order.
        with open(os.path.join(MODELS_DIR, 'feature_config.json')) as f:
            config = json.load(f)

        self.feature_cols = config['feature_cols']  # ordered list of 19 feature names
        self.window_size  = config['window_size']   # 30 days of history per prediction
        self.n_features   = config['n_features']    # 19

        # Scalers were fitted on training data only to prevent data leakage
        self.feature_scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.pkl'))
        self.target_scaler  = joblib.load(os.path.join(MODELS_DIR, 'target_scaler.pkl'))
        self.lr_model       = joblib.load(os.path.join(MODELS_DIR, 'lr_reg.pkl'))

        # Fusion weights from validation grid search: GRU=0.9, LR=0.0, Transformer=0.1
        with open(os.path.join(MODELS_DIR, 'hfm_config.json')) as f:
            hfm_config = json.load(f)

        self.w_gru = hfm_config['weights']['gru']
        self.w_lr  = hfm_config['weights']['lr']
        self.w_tf  = hfm_config['weights']['transformer']

        # Load GRU weights and switch to eval mode — no training happening here
        self.gru = GRUModel(input_size=self.n_features)
        self.gru.load_state_dict(
            torch.load(os.path.join(MODELS_DIR, 'gru_model.pt'), map_location=DEVICE)
        )
        self.gru.to(DEVICE).eval()

        # Load Transformer weights
        self.transformer = TransformerModel(input_size=self.n_features)
        self.transformer.load_state_dict(
            torch.load(os.path.join(MODELS_DIR, 'transformer_model.pt'), map_location=DEVICE)
        )
        self.transformer.to(DEVICE).eval()

        # Load FinBERT once at startup — downloading on every request would be too slow
        self._finbert = None
        try:
            from transformers import pipeline
            self._finbert = pipeline(
                'text-classification',
                model='ProsusAI/finbert',
                return_all_scores=False,
                truncation=True,
                max_length=128
            )
            log.info('FinBERT loaded successfully')
        except Exception as e:
            log.warning('FinBERT unavailable — sentiment will default to 0.0: %s', e)

        self._ready = True
        log.info('All models ready on %s | GRU=%.1f  TF=%.1f  LR=%.1f',
                 DEVICE, self.w_gru, self.w_tf, self.w_lr)

    def _fetch_price_data(self, ticker='BTC-USD', days=90):
        # Downloads recent daily OHLCV candles and appends today's partial candle
        # using intraday data so the model always has the latest price information.
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=days)

        raw = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )

        if raw.empty or len(raw) < self.window_size + 5:
            raise ValueError(f'Not enough historical data available for {ticker}')

        # yfinance sometimes returns MultiIndex columns — flatten to a simple list
        if hasattr(raw.columns, 'levels'):
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

        # fast_info gives the real-time tick price; fall back to last close if it fails
        try:
            live_price = float(yf.Ticker(ticker).fast_info['last_price'])
        except Exception:
            live_price = float(raw['Close'].iloc[-1])

        # Build today's candle from hourly data and append it to the daily history
        try:
            intraday = yf.download(ticker, period='1d', interval='1h', progress=False)
            if hasattr(intraday.columns, 'levels'):
                intraday.columns = [c[0] if isinstance(c, tuple) else c for c in intraday.columns]

            if not intraday.empty:
                today_candle = pd.DataFrame({
                    'Open':   [float(intraday['Open'].iloc[0])],
                    'High':   [float(intraday['High'].max())],
                    'Low':    [float(intraday['Low'].min())],
                    'Close':  [live_price],
                    'Volume': [float(intraday['Volume'].sum())],
                }, index=[pd.Timestamp(datetime.today().date())])

                if today_candle.index[0] not in raw.index:
                    raw = pd.concat([raw, today_candle])
                else:
                    raw.loc[today_candle.index[0], 'Close'] = live_price
        except Exception:
            pass

        # 14-day annualised volatility displayed on the dashboard
        log_returns = np.log(raw['Close'] / raw['Close'].shift(1)).dropna()
        vol_14d     = float(log_returns.tail(14).std() * np.sqrt(365) * 100)

        return raw, live_price, vol_14d

    def _build_features(self, df):
        # Calculates all 19 technical features from raw OHLCV data.
        # Order and clipping values must match the training notebook exactly,
        # otherwise the scaler will produce incorrect scaled values.
        d      = df.copy()
        close  = d['Close']
        volume = d['Volume']

        # Returns clipped at ±15% to reduce the effect of extreme daily moves
        d['Return']     = close.pct_change().clip(-0.15, 0.15)
        d['Log_Return'] = np.log(close / close.shift(1)).clip(-0.15, 0.15)
        d['Return_3']   = close.pct_change(3).clip(-0.15, 0.15)
        d['Return_7']   = close.pct_change(7).clip(-0.15, 0.15)
        d['Return_14']  = close.pct_change(14).clip(-0.15, 0.15)

        # Trend — how far price is sitting from its moving averages
        sma7  = close.rolling(7).mean()
        sma21 = close.rolling(21).mean()
        d['Price_vs_SMA7']  = (close - sma7)  / (sma7  + 1e-9)
        d['Price_vs_SMA21'] = (close - sma21) / (sma21 + 1e-9)

        # MACD — difference between short and long-term exponential averages
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        d['EMA_ratio']   = ema12 / (ema26 + 1e-9)
        d['MACD']        = ema12 - ema26
        d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
        d['MACD_Hist']   = d['MACD'] - d['MACD_Signal']

        # RSI — measures whether the asset is overbought or oversold over 14 days
        # Small epsilon prevents division by zero when average loss is zero
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        d['RSI_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        # Bollinger Bands — show where price sits within its recent volatility range
        sma20    = close.rolling(20).mean()
        std20    = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        d['BB_Width'] = (bb_upper - bb_lower) / (sma20 + 1e-9)
        d['BB_Pos']   = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

        # Volatility — Vol_ratio compares short-term to longer-term volatility
        d['Vol_7']     = d['Log_Return'].rolling(7).std()
        vol30          = d['Log_Return'].rolling(30).std()
        d['Vol_ratio'] = d['Vol_7'] / (vol30 + 1e-9)

        # ATR — normalised average true range captures intraday price swings
        true_range = pd.concat([
            d['High'] - d['Low'],
            (d['High'] - close.shift(1)).abs(),
            (d['Low']  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        d['ATR_14'] = true_range.rolling(14).mean() / (close + 1e-9)

        d['Volume_Change'] = volume.pct_change().clip(-0.15, 0.15)
        d['Sentiment']     = self._get_sentiment()

        d.dropna(inplace=True)
        return d[self.feature_cols].values

    def _get_sentiment(self):
        # Fetches recent financial headlines, scores them with FinBERT,
        # and returns a single value between -1 (very negative) and +1 (very positive).
        # Returns 0.0 if FinBERT is unavailable or no headlines are found.
        if self._finbert is None:
            return 0.0

        try:
            import feedparser

            rss_feeds = [
                'https://feeds.feedburner.com/CoinDesk',
                'https://cryptopanic.com/news/rss/',
                'https://feeds.bbci.co.uk/news/business/rss.xml',
            ]

            headlines = []
            for url in rss_feeds:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    headlines.append(entry.title)
                # Stop after the first feed that returns results
                if headlines:
                    break

            if not headlines:
                return 0.0

            # Map FinBERT labels to numeric values and weight by confidence
            label_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
            results   = self._finbert(headlines[:10])

            scores = [
                label_map.get(
                    r[0]['label'].lower() if isinstance(r, list) else r['label'].lower(), 0.0
                ) * (r[0]['score'] if isinstance(r, list) else r['score'])
                for r in results
            ]

            return float(np.mean(scores))

        except Exception as e:
            log.warning('Sentiment scoring failed: %s', e)
            return 0.0

    def predict(self, ticker='BTC-USD'):
        # Runs the full prediction pipeline — fetches data, engineers features,
        # runs GRU + Transformer + LR, blends using HFM weights, and returns results.
        self.init()

        raw, live_price, vol14 = self._fetch_price_data(ticker)
        features = self._build_features(raw)

        if len(features) < self.window_size:
            raise ValueError('Not enough clean data rows after feature engineering')

        # Scale and extract the most recent 30-day window
        scaled = self.feature_scaler.transform(features)
        window = scaled[-self.window_size:]
        x      = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)

        # No gradients needed during inference
        with torch.no_grad():
            gru_reg, gru_cls = self.gru(x)
            tf_reg,  tf_cls  = self.transformer(x)

        gru_r = float(gru_reg.squeeze().cpu())
        gru_c = float(gru_cls.squeeze().cpu())
        tf_r  = float(tf_reg.squeeze().cpu())
        tf_c  = float(tf_cls.squeeze().cpu())

        # Linear regression takes a flat single-row input rather than a sequence
        lr_input = scaled[-1].reshape(1, -1)
        lr_r     = float(self.lr_model.predict(lr_input)[0])
        lr_c     = 1.0 if lr_r > 0 else 0.0

        # Blend all three models using the HFM weights
        fused_r = self.w_gru * gru_r + self.w_lr * lr_r + self.w_tf * tf_r
        fused_c = self.w_gru * gru_c + self.w_lr * lr_c + self.w_tf * tf_c

        pred_return = float(self.target_scaler.inverse_transform([[fused_r]])[0][0])
        direction   = 'up' if fused_c >= 0.5 else 'down'

        # Align return sign with direction — occasional mismatches happen when
        # the regression and classification heads disagree slightly
        if direction == 'up' and pred_return < 0:
            pred_return = abs(pred_return)
        elif direction == 'down' and pred_return > 0:
            pred_return = -abs(pred_return)

        # Cap at ±10% to prevent extreme outlier predictions reaching the UI
        pred_return = max(min(pred_return, 0.10), -0.10)
        pred_price  = live_price * (1.0 + pred_return)

        # Confidence is always shown from the perspective of the predicted direction
        confidence = fused_c if direction == 'up' else (1.0 - fused_c)

        gru_ret = float(self.target_scaler.inverse_transform([[gru_r]])[0][0])
        gru_dir = 'up' if gru_c >= 0.5 else 'down'

        return {
            'ticker':          ticker,
            'current_price':   round(live_price, 2),
            'predicted_price': round(pred_price, 2),
            'pred_return_pct': round(pred_return * 100, 4),
            'direction':       direction,
            'confidence':      round(confidence, 4),
            'vol_14d':         round(vol14, 4),
            'sentiment_score': round(self._get_sentiment(), 4),
            'models': {
                'gru': {
                    'reg': round(gru_r, 4),
                    'cls': round(gru_c, 4),
                    'direction': gru_dir,
                    'pred_return_pct': round(gru_ret * 100, 4)
                },
                'transformer': {
                    'reg': round(tf_r, 4),
                    'cls': round(tf_c, 4)
                },
                'lr': {
                    'reg': round(lr_r, 4),
                    'cls': round(lr_c, 4)
                },
            },
            'hfm_weights': {
                'gru': self.w_gru,
                'lr':  self.w_lr,
                'tf':  self.w_tf
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }


# Single shared instance — created once when the module is first imported
_loader = ModelLoader()


def get_prediction(ticker='BTC-USD'):
    # Entry point called by the Flask route to get a prediction for a given ticker
    _loader.init()
    return _loader.predict(ticker)
