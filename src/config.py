### config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_config():
    return {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT"),
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD"),
        "DB_DB": os.getenv("DB_DB"),
    }
