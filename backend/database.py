from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    # Stores registered user accounts with role-based access.
    # Role is either 'user' (default) or 'admin'.
    __tablename__ = 'users'

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20),  default='user')
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)
    is_active     = db.Column(db.Boolean,     default=True)

    # One user can have many predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def to_dict(self):
        # Returns a safe dictionary representation — password hash is never included
        return {
            'id':         self.id,
            'username':   self.username,
            'email':      self.email,
            'role':       self.role,
            'created_at': self.created_at.isoformat(),
            'is_active':  self.is_active,
        }


class Prediction(db.Model):
    # Stores every forecast run by a user, including all model output values.
    # Used to populate the prediction history table on the dashboard.
    __tablename__ = 'predictions'

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    ticker          = db.Column(db.String(20),  default='BTC-USD')
    current_price   = db.Column(db.Float,  nullable=False)
    predicted_price = db.Column(db.Float,  nullable=False)
    pred_return_pct = db.Column(db.Float,  nullable=False)
    direction       = db.Column(db.String(10), nullable=False)
    confidence      = db.Column(db.Float,  nullable=False)
    vol_14d         = db.Column(db.Float)
    sentiment_score = db.Column(db.Float,  default=0.0)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':              self.id,
            'ticker':          self.ticker,
            'current_price':   self.current_price,
            'predicted_price': self.predicted_price,
            'pred_return_pct': self.pred_return_pct,
            'direction':       self.direction,
            'confidence':      self.confidence,
            'vol_14d':         self.vol_14d,
            'sentiment_score': self.sentiment_score,
            'created_at':      self.created_at.isoformat(),
        }
