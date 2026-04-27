# standard libs
import os
import json
import math
import logging
from datetime import datetime, timedelta

# data
import numpy as np
import pandas as pd
import joblib

# deep learning
import torch
import torch.nn as nn

# market data
import yfinance as yf


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')


def build_classifier_head(input_size, dropout=0.2):
    # small MLP for direction prediction
    return nn.Sequential(
        nn.Linear(input_size, input_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(input_size, 1),
        nn.Sigmoid()
    )


class GRUModel(nn.Module):
    # sequence model for time series inputs

    def __init__(self, input_size=19, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout  = nn.Dropout(dropout)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = build_classifier_head(hidden_size, dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        last = self.dropout(out[:, -1, :])
        return self.reg_head(last), self.cls_head(last)


class PositionalEncoding(nn.Module):
    # adds positional information for transformer input

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
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    # transformer encoder for sequence modelling

    def __init__(self, input_size=19, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout  = nn.Dropout(dropout)
        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = build_classifier_head(d_model, dropout)

    def forward(self, x):
        x    = self.pos_enc(self.input_proj(x))
        x    = self.encoder(x)
        last = self.dropout(x[:, -1, :])
        return self.reg_head(last), self.cls_head(last)


class ModelLoader:
    # keeps models in memory - avoid reloading every request

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def init(self):
        if self._ready:
            return

        # load feature config
        with open(os.path.join(MODELS_DIR, 'feature_config.json')) as f:
            config = json.load(f)

        self.feature_cols = config['feature_cols']
        self.window_size  = config['window_size']
        self.n_features   = config['n_features']

        # scalers and linear model
        self.feature_scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.pkl'))
        self.target_scaler  = joblib.load(os.path.join(MODELS_DIR, 'target_scaler.pkl'))
        self.lr_model       = joblib.load(os.path.join(MODELS_DIR, 'lr_reg.pkl'))

        # fusion weights from grid search
        with open(os.path.join(MODELS_DIR, 'hfm_config.json')) as f:
            hfm = json.load(f)

        self.w_gru = hfm['weights']['gru']
        self.w_lr  = hfm['weights']['lr']
        self.w_tf  = hfm['weights']['transformer']

        # load GRU
        self.gru = GRUModel(input_size=self.n_features)
        self.gru.load_state_dict(
            torch.load(os.path.join(MODELS_DIR, 'gru_model.pt'), map_location=DEVICE)
        )
        self.gru.to(DEVICE).eval()

        # load Transformer
        self.transformer = TransformerModel(input_size=self.n_features)
        self.transformer.load_state_dict(
            torch.load(os.path.join(MODELS_DIR, 'transformer_model.pt'), map_location=DEVICE)
        )
        self.transformer.to(DEVICE).eval()

        # load FinBERT once at startup
        self._finbert = None
        try:
            from transformers import pipeline
            self._finbert = pipeline(
                'text-classification',
                model='ProsusAI/finbert',
                truncation=True,
                max_length=128
            )
            log.info('FinBERT loaded')
        except Exception as e:
            log.warning('FinBERT unavailable: %s', e)

        self._ready = True
        log.info('Models ready on %s | GRU=%.1f TF=%.1f LR=%.1f',
                 DEVICE, self.w_gru, self.w_tf, self.w_lr)

    def _fetch_price_data(self, ticker='BTC-USD', days=90):
        end   = datetime.today()
        start = end - timedelta(days=days)

        df = yf.download(
            ticker,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            progress=False
        )

        if df.empty or len(df) < self.window_size + 5:
            raise ValueError(f'Not enough data for {ticker}')

        if hasattr(df.columns, 'levels'):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        try:
            live_price = float(yf.Ticker(ticker).fast_info['last_price'])
        except Exception:
            live_price = float(df['Close'].iloc[-1])

        # append today's intraday candle
        try:
            intraday = yf.download(ticker, period='1d', interval='1h', progress=False)
            if hasattr(intraday.columns, 'levels'):
                intraday.columns = [c[0] if isinstance(c, tuple) else c for c in intraday.columns]
            if not intraday.empty:
                today = pd.DataFrame({
                    'Open':   [float(intraday['Open'].iloc[0])],
                    'High':   [float(intraday['High'].max())],
                    'Low':    [float(intraday['Low'].min())],
                    'Close':  [live_price],
                    'Volume': [float(intraday['Volume'].sum())],
                }, index=[pd.Timestamp(datetime.today().date())])
                if today.index[0] not in df.index:
                    df = pd.concat([df, today])
                else:
                    df.loc[today.index[0], 'Close'] = live_price
        except Exception:
            pass

        log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        vol14   = float(log_ret.tail(14).std() * np.sqrt(365) * 100)

        return df, live_price, vol14

    def _build_features(self, df):
        # must match training notebook exactly - same order, same clipping
        d      = df.copy()
        close  = d['Close']
        volume = d['Volume']

        # returns
        d['Return']     = close.pct_change().clip(-0.15, 0.15)
        d['Log_Return'] = np.log(close / close.shift(1)).clip(-0.15, 0.15)
        d['Return_3']   = close.pct_change(3).clip(-0.15, 0.15)
        d['Return_7']   = close.pct_change(7).clip(-0.15, 0.15)
        d['Return_14']  = close.pct_change(14).clip(-0.15, 0.15)

        # trend
        sma7  = close.rolling(7).mean()
        sma21 = close.rolling(21).mean()
        d['Price_vs_SMA7']  = (close - sma7)  / (sma7  + 1e-9)
        d['Price_vs_SMA21'] = (close - sma21) / (sma21 + 1e-9)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        d['EMA_ratio']   = ema12 / (ema26 + 1e-9)
        d['MACD']        = ema12 - ema26
        d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
        d['MACD_Hist']   = d['MACD'] - d['MACD_Signal']

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        d['RSI_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        # Bollinger Bands
        sma20    = close.rolling(20).mean()
        std20    = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        d['BB_Width'] = (bb_upper - bb_lower) / (sma20 + 1e-9)
        d['BB_Pos']   = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

        # volatility
        d['Vol_7']     = d['Log_Return'].rolling(7).std()
        vol30          = d['Log_Return'].rolling(30).std()
        d['Vol_ratio'] = d['Vol_7'] / (vol30 + 1e-9)

        # ATR
        tr = pd.concat([
            d['High'] - d['Low'],
            (d['High'] - close.shift(1)).abs(),
            (d['Low']  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        d['ATR_14'] = tr.rolling(14).mean() / (close + 1e-9)

        d['Volume_Change'] = volume.pct_change().clip(-0.15, 0.15)
        d['Sentiment']     = self._get_sentiment()

        d.dropna(inplace=True)
        return d[self.feature_cols].values

    def _get_sentiment(self):
        if self._finbert is None:
            return 0.0

        try:
            import feedparser

            feeds = [
                'https://feeds.feedburner.com/CoinDesk',
                'https://cryptopanic.com/news/rss/',
                'https://feeds.bbci.co.uk/news/business/rss.xml',
            ]

            headlines = []
            for url in feeds:
                feed = feedparser.parse(url)
                headlines = [e.title for e in feed.entries[:5]]
                if headlines:
                    break

            if not headlines:
                return 0.0

            label_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
            results   = self._finbert(headlines[:10])
            scores    = [
                label_map.get(r['label'].lower(), 0.0) * r['score']
                for r in results
            ]
            return float(np.mean(scores))

        except Exception as e:
            log.warning('Sentiment failed: %s', e)
            return 0.0

    def predict(self, ticker='BTC-USD'):
        self.init()

        df, price, vol = self._fetch_price_data(ticker)
        features = self._build_features(df)

        if len(features) < self.window_size:
            raise ValueError('Not enough data')

        scaled = self.feature_scaler.transform(features)
        window = scaled[-self.window_size:]
        x      = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            gru_r, gru_c = self.gru(x)
            tf_r,  tf_c  = self.transformer(x)

        gru_r = float(gru_r.squeeze().cpu())
        gru_c = float(gru_c.squeeze().cpu())
        tf_r  = float(tf_r.squeeze().cpu())
        tf_c  = float(tf_c.squeeze().cpu())

        lr_input = scaled[-1].reshape(1, -1)
        lr_r     = float(self.lr_model.predict(lr_input)[0])
        lr_c     = 1.0 if lr_r > 0 else 0.0

        # blend using HFM weights
        fused_r = self.w_gru * gru_r + self.w_tf * tf_r + self.w_lr * lr_r
        fused_c = self.w_gru * gru_c + self.w_tf * tf_c + self.w_lr * lr_c

        pred_return = float(self.target_scaler.inverse_transform([[fused_r]])[0][0])
        direction   = 'up' if fused_c >= 0.5 else 'down'

        # align sign with direction
        if direction == 'up' and pred_return < 0:
            pred_return = abs(pred_return)
        elif direction == 'down' and pred_return > 0:
            pred_return = -abs(pred_return)

        pred_return = max(min(pred_return, 0.10), -0.10)
        pred_price  = price * (1.0 + pred_return)
        confidence  = fused_c if direction == 'up' else (1.0 - fused_c)

        gru_ret = float(self.target_scaler.inverse_transform([[gru_r]])[0][0])
        gru_dir = 'up' if gru_c >= 0.5 else 'down'

        return {
            'ticker':          ticker,
            'current_price':   round(price, 2),
            'predicted_price': round(pred_price, 2),
            'pred_return_pct': round(pred_return * 100, 4),
            'direction':       direction,
            'confidence':      round(confidence, 4),
            'vol_14d':         round(vol, 4),
            'sentiment_score': round(self._get_sentiment(), 4),
            'models': {
                'gru': {
                    'reg':             round(gru_r, 4),
                    'cls':             round(gru_c, 4),
                    'direction':       gru_dir,
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


_loader = ModelLoader()


def get_prediction(ticker='BTC-USD'):
    _loader.init()
    return _loader.predict(ticker)