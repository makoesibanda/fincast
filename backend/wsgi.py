import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app  # Your original line stays below


from app import create_app

# Entry point for production deployment via Gunicorn
application = create_app()
