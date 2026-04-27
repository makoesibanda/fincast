import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    # signs and verifies JWT tokens - override with a strong key in production
    SECRET_KEY = os.getenv('SECRET_KEY', 'simon_sibanda_fincast_2026')

    # falls back to SQLite if DATABASE_URL is not set
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///fincast.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # default asset if no ticker is passed in the request
    DEFAULT_TICKER = os.getenv('DEFAULT_TICKER', 'BTC-USD')