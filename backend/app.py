import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from config import Config
from database import db

# Blueprints for different parts of the system
# keeps routes modular instead of putting everything in one file
from routes.auth import auth_bp
from routes.forecast import forecast_bp
from routes.sentiment import sentiment_bp
from routes.admin import admin_bp


# Path to frontend folder (so Flask can serve the UI directly)
FRONTEND = os.path.join(os.path.dirname(__file__), '..', 'frontend')


def create_app(cfg=Config):
    # app factory

    app = Flask(__name__, static_folder=None)
    app.config.from_object(cfg)

    # allow frontend (JS) to talk to backend
    CORS(app)

    # initialise database
    db.init_app(app)

    # register API routes
    app.register_blueprint(auth_bp,      url_prefix='/api/auth')
    app.register_blueprint(forecast_bp,  url_prefix='/api/forecast')
    app.register_blueprint(sentiment_bp, url_prefix='/api/sentiment')
    app.register_blueprint(admin_bp,     url_prefix='/api/admin')

    # serve main page
    @app.route('/')
    def index():
        return send_from_directory(FRONTEND, 'index.html')

    # serve other frontend files (css, js, html pages)
    @app.route('/<path:filename>')
    def frontend(filename):
        return send_from_directory(FRONTEND, filename)

    # create tables if they don't exist yet
    with app.app_context():
        db.create_all()

        # create default admin on first run
        from werkzeug.security import generate_password_hash
        from database import User
        if not User.query.filter_by(role='admin').first():
            admin = User(
                username='fincast',
                email='admin1@fincast.com',
                password_hash=generate_password_hash('admin1234'),
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()

    return app


if __name__ == '__main__':
    # run locally
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)