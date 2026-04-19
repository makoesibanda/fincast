from app import create_app

# Entry point for production deployment via Gunicorn
application = create_app()
