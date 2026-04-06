from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    env_path = Path(__file__).resolve().parent.parent.parent / "settings" / ".env"
    load_dotenv(env_path)