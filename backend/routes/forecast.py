from flask import Blueprint, jsonify, request, current_app, g
from routes.auth import login_required
from database import db, Prediction
from model_loader import get_prediction

# yfinance and datetime are imported inside history_chart to keep the top-level
# imports lightweight — they're only needed for that one route
import yfinance as yf
from datetime import datetime, timedelta

forecast_bp = Blueprint('forecast', __name__)


@forecast_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    # Runs the HFM prediction pipeline and saves the result to the database.
    # Accepts the ticker via POST body or GET query parameter.
    ticker = 'BTC-USD'
    if request.method == 'POST':
        data   = request.get_json(silent=True) or {}
        ticker = data.get('ticker', 'BTC-USD').upper()
    else:
        ticker = request.args.get('ticker', 'BTC-USD').upper()

    try:
        result = get_prediction(ticker)
    except Exception as e:
        current_app.logger.error('Prediction error: %s', e)
        return jsonify({'error': str(e)}), 500

    # Save the prediction to the database — log a warning if it fails but still return the result
    try:
        row = Prediction(
            user_id         = g.current_user.id,
            ticker          = result['ticker'],
            current_price   = result['current_price'],
            predicted_price = result['predicted_price'],
            pred_return_pct = result['pred_return_pct'],
            direction       = result['direction'],
            confidence      = result['confidence'],
            vol_14d         = result['vol_14d'],
            sentiment_score = result.get('sentiment_score', 0.0),
        )
        db.session.add(row)
        db.session.commit()
        result['prediction_id'] = row.id
    except Exception as e:
        current_app.logger.warning('DB save failed: %s', e)
        db.session.rollback()

    return jsonify(result), 200


@forecast_bp.route('/history', methods=['GET'])
@login_required
def history():
    # Returns the most recent predictions for the logged in user, newest first
    limit = min(int(request.args.get('limit', 20)), 100)
    rows  = (Prediction.query
             .filter_by(user_id=g.current_user.id)
             .order_by(Prediction.created_at.desc())
             .limit(limit).all())
    return jsonify([r.to_dict() for r in rows]), 200


@forecast_bp.route('/models', methods=['GET'])
@login_required
def model_stats():
    # Returns the test set results from the v2 notebook evaluation.
    # These are static values from the 2024 held-out test set — not recalculated on each request.
    stats = [
        {'model': 'Linear Regression', 'sharpe':  1.171, 'dir_acc': 53.30, 'max_dd': -33.70, 'mae': 2.0212, 'hit_rate': 53.20},
        {'model': 'Random Forest',     'sharpe':  0.495, 'dir_acc': 48.90, 'max_dd': -44.70, 'mae': 2.0887, 'hit_rate': 48.80},
        {'model': 'XGBoost',           'sharpe': -0.323, 'dir_acc': 47.80, 'max_dd': -54.70, 'mae': 2.3830, 'hit_rate': 47.70},
        {'model': 'LSTM',              'sharpe':  1.834, 'dir_acc': 52.69, 'max_dd': -26.20, 'mae': 2.0417, 'hit_rate': 52.60},
        {'model': 'GRU',               'sharpe':  2.612, 'dir_acc': 54.19, 'max_dd': -18.10, 'mae': 2.0305, 'hit_rate': 54.10},
        {'model': '1D-CNN',            'sharpe':  1.834, 'dir_acc': 52.69, 'max_dd': -26.20, 'mae': 2.0424, 'hit_rate': 52.60},
        {'model': 'Transformer',       'sharpe':  1.834, 'dir_acc': 52.69, 'max_dd': -26.20, 'mae': 2.0414, 'hit_rate': 52.60},
        {'model': 'Hybrid LSTM+TF',    'sharpe':  1.834, 'dir_acc': 52.69, 'max_dd': -26.20, 'mae': 2.0453, 'hit_rate': 52.60},
        {'model': 'Multimodal Hybrid', 'sharpe':  1.834, 'dir_acc': 52.69, 'max_dd': -26.20, 'mae': 2.0423, 'hit_rate': 52.60},
        {'model': 'HFM (Fusion)',      'sharpe':  2.209, 'dir_acc': 52.40, 'max_dd': -18.10, 'mae': 2.0313, 'hit_rate': 52.30},
    ]

    walk_forward = [
        {'window': 1, 'dir_acc': 56.8, 'sharpe':  2.948, 'max_dd': -20.30},
        {'window': 2, 'dir_acc': 46.8, 'sharpe': -1.022, 'max_dd': -24.10},
        {'window': 3, 'dir_acc': 54.5, 'sharpe':  3.511, 'max_dd': -12.70},
    ]

    # Top 5 features by SHAP importance from GradientExplainer on the test set
    shap_features = [
        {'feature': 'Return_14',    'rank': 1},
        {'feature': 'MACD_Signal',  'rank': 2},
        {'feature': 'MACD_Hist',    'rank': 3},
        {'feature': 'Volume_Change','rank': 4},
        {'feature': 'Return_7',     'rank': 5},
    ]

    return jsonify({
        'model_results': stats,
        'walk_forward':  walk_forward,
        'shap_top5':     shap_features,
        'primary_model': 'GRU',
        'best_sharpe':   2.612,
        'best_dir_acc':  54.19,
    }), 200


@forecast_bp.route('/history_chart', methods=['GET'])
@login_required
def history_chart():
    # Fetches historical daily close prices for the price chart on the dashboard (FR2).
    # Returns dates and prices as parallel arrays for Chart.js to consume.
    ticker = request.args.get('ticker', 'BTC-USD').upper()
    days   = min(int(request.args.get('days', 60)), 365)

    try:
        end   = datetime.today()
        start = end - timedelta(days=days)

        raw = yf.download(
            ticker,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            progress=False
        )

        if raw.empty:
            return jsonify({'dates': [], 'prices': []}), 200

        # Flatten MultiIndex columns if yfinance returns them
        if hasattr(raw.columns, 'levels'):
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

        dates  = [str(d.date()) for d in raw.index]
        prices = [round(float(p), 2) for p in raw['Close']]

        return jsonify({'dates': dates, 'prices': prices, 'ticker': ticker}), 200

    except Exception as e:
        current_app.logger.error('Chart data error: %s', e)
        return jsonify({'dates': [], 'prices': [], 'error': str(e)}), 200
