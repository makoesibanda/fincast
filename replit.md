# FinCast — Bitcoin Forecasting Platform

## Project Overview
FinCast is an intelligent Bitcoin (BTC) forecasting platform that uses machine learning and deep learning models to predict daily cryptocurrency returns and market direction. Built as a Goldsmiths BSc Computer Science final-year project.

## Tech Stack
- **Backend:** Python 3.12 + Flask 3.0.3 (serves both API and static frontend)
- **Frontend:** Vanilla HTML/CSS/JavaScript (SPA-style, served from `/frontend/`)
- **Database:** PostgreSQL (Replit-managed, via `DATABASE_URL` env var)
- **ML/AI:** PyTorch (CPU-only), Scikit-learn, XGBoost, Transformers (HuggingFace FinBERT)
- **Data:** yfinance for market data, feedparser for news RSS
- **Auth:** JWT (PyJWT)
- **WSGI:** Gunicorn (for production)

## Project Structure
```
backend/
  app.py              # Flask app factory + static file serving
  config.py           # Config from env vars
  database.py         # SQLAlchemy models (User, Prediction)
  model_loader.py     # ML model classes (GRU, Transformer, HFM ensemble)
  wsgi.py             # WSGI entry point for Gunicorn
  routes/
    auth.py           # /api/auth/* (register, login, me)
    forecast.py       # /api/forecast/* (predict, history, models)
    sentiment.py      # /api/sentiment/*
    admin.py          # /api/admin/*
  models/saved/       # Saved model weights (.pt, .pkl) and config (.json)
frontend/
  index.html          # Landing / Login / Register
  dashboard.html      # Main user interface
  models.html         # Model performance metrics
  sentiment.html      # Sentiment analysis view
  admin.html          # Admin management interface
```

## Environment Variables
- `SECRET_KEY` — Flask secret key for JWT signing
- `DATABASE_URL` — PostgreSQL connection string (runtime-managed by Replit)
- `DEFAULT_TICKER` — Default crypto ticker, default `BTC-USD`

## Running the App
- **Development:** `cd backend && python3 app.py` (port 5000, host 0.0.0.0)
- **Production:** `gunicorn --bind=0.0.0.0:5000 --reuse-port --chdir=backend wsgi:application`

## Workflow
- Workflow: "Start application" — runs `cd backend && python3 app.py` on port 5000

## Key Notes
- PyTorch is installed as CPU-only (GPU version too large for Replit disk)
- The app uses Replit's managed PostgreSQL via `DATABASE_URL` env var
- HFM (Hybrid Fusion Model) combines GRU (90%) + Transformer (10%) + Linear Regression
