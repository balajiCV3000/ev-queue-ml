import os
from dotenv import load_dotenv

load_dotenv()

# Google Maps API key (required for route generation)
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("MAPS_API_KEY", "")

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("DEBUG", "false").strip().lower() == "true"
BOOTSTRAP_SIMULATION = os.getenv("BOOTSTRAP_SIMULATION", "true").strip().lower() == "true"
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
REQUIRE_MAPS_API_KEY = os.getenv(
    "REQUIRE_MAPS_API_KEY",
    "true" if APP_ENV == "production" else "false",
).strip().lower() == "true"

# Simulation parameters
TIME_STEP_SECONDS = int(os.getenv("TIME_STEP_SECONDS", "60"))
CHARGE_THRESHOLD = float(os.getenv("CHARGE_THRESHOLD", "0.2"))
OPTIMIZATION_INTERVAL = int(os.getenv("OPTIMIZATION_INTERVAL", "10"))


def validate_required_config():
    if REQUIRE_MAPS_API_KEY and not GOOGLE_MAPS_API_KEY:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY (or MAPS_API_KEY) is required when "
            "REQUIRE_MAPS_API_KEY=true."
        )
