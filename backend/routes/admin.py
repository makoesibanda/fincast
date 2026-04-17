import jwt
from functools import wraps
from flask import Blueprint, jsonify, request, current_app, g
from sqlalchemy import func
from database import db, User, Prediction

admin_bp = Blueprint('admin', __name__)


def admin_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        try:
            payload = jwt.decode(auth.split(' ', 1)[1],
                                 current_app.config['SECRET_KEY'],
                                 algorithms=['HS256'])
        except Exception:
            return jsonify({'error': 'Invalid token'}), 401
        user = User.query.get(payload['user_id'])
        if not user or not user.is_active:
            return jsonify({'error': 'Account not found'}), 401
        if user.role != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        g.current_user = user
        return f(*args, **kwargs)
    return wrapped


@admin_bp.route('/users', methods=['GET'])
@admin_required
def list_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify([u.to_dict() for u in users]), 200


@admin_bp.route('/users/<int:uid>', methods=['GET', 'PATCH'])
@admin_required
def manage_user(uid):
    user = User.query.get_or_404(uid)
    if request.method == 'GET':
        return jsonify(user.to_dict()), 200
    data = request.get_json(silent=True) or {}
    if 'is_active' in data:
        if user.id == g.current_user.id:
            return jsonify({'error': 'Cannot deactivate own account'}), 400
        user.is_active = bool(data['is_active'])
    if 'role' in data and data['role'] in ('user', 'admin'):
        user.role = data['role']
    db.session.commit()
    return jsonify(user.to_dict()), 200


@admin_bp.route('/predictions', methods=['GET'])
@admin_required
def list_predictions():
    page = max(int(request.args.get('page', 1)), 1)
    per  = min(int(request.args.get('per_page', 20)), 100)
    pag  = (Prediction.query.order_by(Prediction.created_at.desc())
            .paginate(page=page, per_page=per, error_out=False))
    return jsonify({'predictions': [p.to_dict() for p in pag.items],
                    'total': pag.total, 'pages': pag.pages, 'page': pag.page}), 200


@admin_bp.route('/stats', methods=['GET'])
@admin_required
def stats():
    total_users  = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    total_preds  = Prediction.query.count()
    up_preds     = Prediction.query.filter_by(direction='up').count()
    down_preds   = Prediction.query.filter_by(direction='down').count()
    avg_conf     = db.session.query(func.avg(Prediction.confidence)).scalar() or 0.0
    return jsonify({
        'users':           {'total': total_users,  'active': active_users},
        'predictions':     {'total': total_preds,  'up': up_preds, 'down': down_preds},
        'avg_confidence':  round(float(avg_conf), 4),
    }), 200
