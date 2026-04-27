import jwt
import datetime
from flask import Blueprint, jsonify, request, current_app, g
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from database import db, User

auth_bp = Blueprint('auth', __name__)


def _decode_token(token):

    # returns None if invalid or expired
    try:
        return jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
    except Exception:
        return None


def login_required(f): 

    # JWT auth decorator
    @wraps(f)
    def wrapped(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401

        payload = _decode_token(auth.split(' ', 1)[1])
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401

        user = User.query.get(payload['user_id'])
        if not user or not user.is_active:
            return jsonify({'error': 'Account not found'}), 401

        g.current_user = user
        return f(*args, **kwargs)
    return wrapped


@auth_bp.route('/register', methods=['POST'])
def register():

    # register new user
    data     = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    email    = (data.get('email')    or '').strip().lower()
    password = data.get('password', '')

    if not username or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken'}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password)
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'Account created', 'user': user.to_dict()}), 201


@auth_bp.route('/login', methods=['POST'])
def login():

    # login and return JWT
    data     = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password', '')

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid credentials'}), 401
    if not user.is_active:
        return jsonify({'error': 'Account disabled'}), 403

    token = jwt.encode(
        {
            'user_id': user.id,
            'role':    user.role,
            'exp':     datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        },
        current_app.config['SECRET_KEY'],
        algorithm='HS256'
    )
    return jsonify({'token': token, 'user': user.to_dict()}), 200


@auth_bp.route('/me', methods=['GET'])
@login_required
def me():
    # current user info
    return jsonify(g.current_user.to_dict()), 200
