import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class Config:
    # Secret key used for signing JWT tokens — must be set in production
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-this-in-production')

    # Database connection — defaults to a local SQLite file for development
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///fincast.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Default ticker used when none is specified in a request
    DEFAULT_TICKER = os.getenv('DEFAULT_TICKER', 'BTC-USD')
