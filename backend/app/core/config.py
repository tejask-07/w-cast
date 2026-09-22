import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    app_name: str = os.getenv("APP_NAME", "W-CAST API")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    cors_origins: list[str] = [
        "http://localhost:5173",
    ]


settings = Settings()
