import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from config import Config
from database import db
from routes.auth      import auth_bp
from routes.forecast  import forecast_bp
from routes.sentiment import sentiment_bp
from routes.admin     import admin_bp

FRONTEND = os.path.join(os.path.dirname(__file__), '..', 'frontend')

def create_app(cfg=Config):
    app = Flask(__name__, static_folder=None)
    app.config.from_object(cfg)

    CORS(app)
    db.init_app(app)

    app.register_blueprint(auth_bp,      url_prefix='/api/auth')
    app.register_blueprint(forecast_bp,  url_prefix='/api/forecast')
    app.register_blueprint(sentiment_bp, url_prefix='/api/sentiment')
    app.register_blueprint(admin_bp,     url_prefix='/api/admin')

    @app.route('/')
    def index():
        return send_from_directory(FRONTEND, 'index.html')

    @app.route('/<path:filename>')
    def frontend(filename):
        return send_from_directory(FRONTEND, filename)

    with app.app_context():
        db.create_all()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
