from flask import Blueprint, jsonify, request, current_app
import numpy as np

sentiment_bp = Blueprint('sentiment', __name__)

# RSS feeds used to pull financial headlines for sentiment scoring
_FEEDS = [
    'https://feeds.feedburner.com/CoinDesk',
    'https://cryptopanic.com/news/rss/',
    'https://feeds.bbci.co.uk/news/business/rss.xml',
]

# Maps FinBERT output labels to numeric sentiment values
_LABEL = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}


def _get_pipe():
    # Loads the FinBERT pipeline for text classification.
    # Returns None if the transformers library is unavailable.
    try:
        from transformers import pipeline
        return pipeline(
            'text-classification',
            model='ProsusAI/finbert',
            return_all_scores=False,
            truncation=True,
            max_length=128
        )
    except Exception as e:
        current_app.logger.warning('FinBERT unavailable: %s', e)
        return None


@sentiment_bp.route('/live', methods=['GET'])
def live():
    # Fetches headlines from the RSS feeds and scores each one with FinBERT.
    # Returns the scored headlines and the mean sentiment across all of them.
    max_items = min(int(request.args.get('max', 15)), 30)

    try:
        import feedparser
    except ImportError:
        return jsonify({'error': 'feedparser not installed'}), 500

    # Collect headlines from feeds until we have enough
    headlines = []
    for url in _FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                headlines.append({
                    'title':  entry.get('title', ''),
                    'source': url.split('/')[2],
                    'link':   entry.get('link', ''),
                })
        except Exception:
            pass
        if len(headlines) >= max_items:
            break

    if not headlines:
        return jsonify({'error': 'No headlines retrieved'}), 503

    pipe = _get_pipe()
    if pipe is None:
        # Return the headlines without scores if FinBERT is unavailable
        return jsonify({
            'headlines':  headlines,
            'mean_score': 0.0,
            'warning':    'FinBERT unavailable'
        }), 200

    # Score each headline and attach label, confidence, and sentiment value
    texts  = [h['title'] for h in headlines]
    raw    = pipe(texts[:max_items])
    scored = []
    for h, r in zip(headlines, raw):
        label = (r[0]['label'] if isinstance(r, list) else r['label']).lower()
        score = (r[0]['score'] if isinstance(r, list) else r['score'])
        val   = _LABEL.get(label, 0.0) * score
        scored.append({
            **h,
            'label':     label,
            'score':     round(score, 4),
            'sentiment': round(val, 4)
        })

    mean = float(np.mean([s['sentiment'] for s in scored])) if scored else 0.0
    return jsonify({
        'headlines':  scored,
        'mean_score': round(mean, 4),
        'count':      len(scored)
    }), 200


@sentiment_bp.route('/score', methods=['POST'])
def score():
    # Scores a list of custom text inputs with FinBERT.
    # Useful for testing or scoring headlines from other sources.
    data  = request.get_json(silent=True) or {}
    texts = data.get('texts', [])

    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return jsonify({'error': 'Provide texts list'}), 400

    pipe = _get_pipe()
    if pipe is None:
        return jsonify({'error': 'FinBERT unavailable'}), 503

    raw     = pipe(texts[:20])
    results = []
    for text, r in zip(texts, raw):
        label = (r[0]['label'] if isinstance(r, list) else r['label']).lower()
        score = (r[0]['score'] if isinstance(r, list) else r['score'])
        results.append({
            'text':      text,
            'label':     label,
            'score':     round(score, 4),
            'sentiment': round(_LABEL.get(label, 0.0) * score, 4)
        })

    return jsonify({'results': results}), 200
