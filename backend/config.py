import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY               = os.getenv('SECRET_KEY', 'change-this-in-production')
    SQLALCHEMY_DATABASE_URI  = os.getenv('DATABASE_URL', 'sqlite:///fincast.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEFAULT_TICKER           = os.getenv('DEFAULT_TICKER', 'BTC-USD')
