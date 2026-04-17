import os, json, math, logging
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import yfinance as yf
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVED  = os.path.join(_DIR, 'models', 'saved')


# ── classifier head used in all v2 DL models ─────────────────────────────────
def _cls_head(n, drop=0.2):
    return nn.Sequential(
        nn.Linear(n, n), nn.ReLU(), nn.Dropout(drop),
        nn.Linear(n, 1), nn.Sigmoid()
    )


# ── model architectures (must match notebook exactly) ────────────────────────

class GRUModel(nn.Module):
    def __init__(self, input_size=19, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru      = nn.GRU(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout  = nn.Dropout(dropout)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = _cls_head(hidden_size, dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        last   = self.dropout(out[:, -1, :])
        return self.reg_head(last), self.cls_head(last)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    def __init__(self, input_size=19, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout    = nn.Dropout(dropout)
        self.reg_head   = nn.Linear(d_model, 1)
        self.cls_head   = _cls_head(d_model, dropout)

    def forward(self, x):
        x    = self.pos_enc(self.input_proj(x))
        x    = self.encoder(x)
        last = self.dropout(x[:, -1, :])
        return self.reg_head(last), self.cls_head(last)


# ── singleton loader ──────────────────────────────────────────────────────────

class _Loader:
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
            cls._inst._ready = False
        return cls._inst

    def init(self):
        if self._ready:
            return

        with open(os.path.join(SAVED, 'feature_config.json')) as f:
            cfg = json.load(f)

        self.feature_cols = cfg['feature_cols']   # 19 cols
        self.window_size  = cfg['window_size']    # 30
        self.n_features   = cfg['n_features']     # 19

        self.feat_scaler = joblib.load(os.path.join(SAVED, 'feature_scaler.pkl'))
        self.tgt_scaler  = joblib.load(os.path.join(SAVED, 'target_scaler.pkl'))
        self.lr_model    = joblib.load(os.path.join(SAVED, 'lr_reg.pkl'))

        # HFM weights from grid search on validation set
        with open(os.path.join(SAVED, 'hfm_config.json')) as f:
            hfm = json.load(f)
        self.w_gru = hfm['weights']['gru']         # 0.9
        self.w_lr  = hfm['weights']['lr']          # 0.0
        self.w_tf  = hfm['weights']['transformer'] # 0.1

        # GRU — primary model
        self.gru = GRUModel(input_size=self.n_features)
        self.gru.load_state_dict(torch.load(
            os.path.join(SAVED, 'gru_model.pt'), map_location=DEVICE))
        self.gru.to(DEVICE).eval()

        # Transformer — secondary (10% weight in HFM)
        self.transformer = TransformerModel(input_size=self.n_features)
        self.transformer.load_state_dict(torch.load(
            os.path.join(SAVED, 'transformer_model.pt'), map_location=DEVICE))
        self.transformer.to(DEVICE).eval()

        # Cache FinBERT so it doesn't reload every request
        self._finbert = None
        try:
            from transformers import pipeline
            self._finbert = pipeline(
                'text-classification', model='ProsusAI/finbert',
                return_all_scores=False, truncation=True, max_length=128)
            log.info('FinBERT cached at startup')
        except Exception as e:
            log.warning('FinBERT unavailable: %s', e)

        self._ready = True
        log.info('Models ready on %s  |  GRU=%.1f  TF=%.1f  LR=%.1f',
                 DEVICE, self.w_gru, self.w_tf, self.w_lr)

    # ── live data ─────────────────────────────────────────────────────────────

    def _fetch(self, ticker='BTC-USD', days=90):
        end   = datetime.today()
        start = end - timedelta(days=days)
        raw   = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                            end=end.strftime('%Y-%m-%d'), progress=False)
        if raw.empty or len(raw) < self.window_size + 5:
            raise ValueError(f'Not enough data for {ticker}')
        if hasattr(raw.columns, 'levels'):
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

        # Live tick price
        try:
            live = float(yf.Ticker(ticker).fast_info['last_price'])
        except Exception:
            live = float(raw['Close'].iloc[-1])

        # Append today's intraday candle
        try:
            intra = yf.download(ticker, period='1d', interval='1h', progress=False)
            if hasattr(intra.columns, 'levels'):
                intra.columns = [c[0] if isinstance(c, tuple) else c for c in intra.columns]
            if not intra.empty:
                today = pd.DataFrame({
                    'Open':   [float(intra['Open'].iloc[0])],
                    'High':   [float(intra['High'].max())],
                    'Low':    [float(intra['Low'].min())],
                    'Close':  [live],
                    'Volume': [float(intra['Volume'].sum())],
                }, index=[pd.Timestamp(datetime.today().date())])
                if today.index[0] not in raw.index:
                    raw = pd.concat([raw, today])
                else:
                    raw.loc[today.index[0], 'Close'] = live
        except Exception:
            pass

        log_ret = np.log(raw['Close'] / raw['Close'].shift(1)).dropna()
        vol14   = float(log_ret.tail(14).std() * np.sqrt(365) * 100)
        return raw, live, vol14

    # ── feature engineering (exact match to notebook Section 4) ──────────────

    def _engineer(self, df):
        d = df.copy()
        c = d['Close']
        v = d['Volume']

        d['Return']       = c.pct_change().clip(-0.15, 0.15)
        d['Log_Return']   = np.log(c / c.shift(1)).clip(-0.15, 0.15)
        d['Return_3']     = c.pct_change(3).clip(-0.15, 0.15)
        d['Return_7']     = c.pct_change(7).clip(-0.15, 0.15)
        d['Return_14']    = c.pct_change(14).clip(-0.15, 0.15)

        sma7  = c.rolling(7).mean()
        sma21 = c.rolling(21).mean()
        d['Price_vs_SMA7']  = (c - sma7)  / (sma7  + 1e-9)
        d['Price_vs_SMA21'] = (c - sma21) / (sma21 + 1e-9)

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        d['EMA_ratio']   = ema12 / (ema26 + 1e-9)
        d['MACD']        = ema12 - ema26
        d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
        d['MACD_Hist']   = d['MACD'] - d['MACD_Signal']

        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        d['RSI_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        sma20  = c.rolling(20).mean()
        std20  = c.rolling(20).std()
        bb_up  = sma20 + 2 * std20
        bb_low = sma20 - 2 * std20
        d['BB_Width'] = (bb_up - bb_low) / (sma20 + 1e-9)
        d['BB_Pos']   = (c - bb_low)     / (bb_up - bb_low + 1e-9)

        d['Vol_7']    = d['Log_Return'].rolling(7).std()
        vol30         = d['Log_Return'].rolling(30).std()
        d['Vol_ratio'] = d['Vol_7'] / (vol30 + 1e-9)

        tr = pd.concat([
            d['High'] - d['Low'],
            (d['High'] - c.shift(1)).abs(),
            (d['Low']  - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        d['ATR_14'] = tr.rolling(14).mean() / (c + 1e-9)

        d['Volume_Change'] = v.pct_change().clip(-0.15, 0.15)
        d['Sentiment']     = self._sentiment()

        d.dropna(inplace=True)
        return d[self.feature_cols].values

    def _sentiment(self):
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
                for e in feed.entries[:5]:
                    headlines.append(e.title)
                if headlines:
                    break
            if not headlines:
                return 0.0
            lmap = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
            results = self._finbert(headlines[:10])
            scores  = [lmap.get(r[0]['label'].lower() if isinstance(r, list)
                                else r['label'].lower(), 0.0) *
                       (r[0]['score'] if isinstance(r, list) else r['score'])
                       for r in results]
            return float(np.mean(scores))
        except Exception as e:
            log.warning('Sentiment failed: %s', e)
            return 0.0

    # ── HFM inference ─────────────────────────────────────────────────────────

    def predict(self, ticker='BTC-USD'):
        self.init()

        raw, live_price, vol14 = self._fetch(ticker)
        feats = self._engineer(raw)

        if len(feats) < self.window_size:
            raise ValueError('Not enough clean rows after feature engineering')

        scaled = self.feat_scaler.transform(feats)
        window = scaled[-self.window_size:]
        x      = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            gru_reg, gru_cls   = self.gru(x)
            tf_reg,  tf_cls    = self.transformer(x)

        gru_r = float(gru_reg.squeeze().cpu())
        gru_c = float(gru_cls.squeeze().cpu())
        tf_r  = float(tf_reg.squeeze().cpu())
        tf_c  = float(tf_cls.squeeze().cpu())

        # LR needs flat input (last window flattened, or last single row)
        lr_input = scaled[-1].reshape(1, -1)
        lr_r     = float(self.lr_model.predict(lr_input)[0])
        lr_c     = 1.0 if lr_r > 0 else 0.0

        # HFM fusion (weights from validation grid search)
        fused_r = self.w_gru * gru_r + self.w_lr * lr_r + self.w_tf * tf_r
        fused_c = self.w_gru * gru_c + self.w_lr * lr_c + self.w_tf * tf_c

        pred_return = float(self.tgt_scaler.inverse_transform([[fused_r]])[0][0])
        direction   = 'up' if fused_c >= 0.5 else 'down'

        if direction == 'up'   and pred_return < 0:
            pred_return = abs(pred_return)
        elif direction == 'down' and pred_return > 0:
            pred_return = -abs(pred_return)

        pred_return = max(min(pred_return, 0.10), -0.10)
        pred_price  = live_price * (1.0 + pred_return)
        confidence  = fused_c if direction == 'up' else (1.0 - fused_c)

        # individual model outputs for comparison panel
        gru_ret  = float(self.tgt_scaler.inverse_transform([[gru_r]])[0][0])
        gru_dir  = 'up' if gru_c >= 0.5 else 'down'

        return {
            'ticker':          ticker,
            'current_price':   round(live_price, 2),
            'predicted_price': round(pred_price, 2),
            'pred_return_pct': round(pred_return * 100, 4),
            'direction':       direction,
            'confidence':      round(confidence, 4),
            'vol_14d':         round(vol14, 4),
            'sentiment_score': round(self._sentiment(), 4),
            'models': {
                'gru':         {'reg': round(gru_r, 4), 'cls': round(gru_c, 4),
                                'direction': gru_dir,
                                'pred_return_pct': round(gru_ret * 100, 4)},
                'transformer': {'reg': round(tf_r, 4),  'cls': round(tf_c, 4)},
                'lr':          {'reg': round(lr_r, 4),  'cls': round(lr_c, 4)},
            },
            'hfm_weights':     {'gru': self.w_gru, 'lr': self.w_lr, 'tf': self.w_tf},
            'timestamp':       datetime.utcnow().isoformat() + 'Z',
        }


_loader = _Loader()

def get_prediction(ticker='BTC-USD'):
    _loader.init()
    return _loader.predict(ticker)
