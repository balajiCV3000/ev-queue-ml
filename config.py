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

# Assignment policy configuration
ASSIGNMENT_POLICY = os.getenv("ASSIGNMENT_POLICY", "greedy")
RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "artifacts/policy_dqn.npz")
RL_NORM_STATS_PATH = os.getenv("RL_NORM_STATS_PATH", "artifacts/norm_stats.json")
ENABLE_STALL_RESCUE = os.getenv("ENABLE_STALL_RESCUE", "false").strip().lower() == "true"
SIMULATE_TRAVEL_ENERGY = os.getenv("SIMULATE_TRAVEL_ENERGY", "true").strip().lower() == "true"
# When enabled, an EV assigned to a station incurs a simulated travel delay
# (distance / TRAVEL_SPEED_KMH) before joining the station queue, instead of
# being teleported into the queue instantly. Defaults to off to preserve
# existing simulation/test behavior; evaluation runs should opt in.
SIMULATE_TRAVEL_TIME = os.getenv("SIMULATE_TRAVEL_TIME", "false").strip().lower() == "true"
TRAVEL_SPEED_KMH = float(os.getenv("TRAVEL_SPEED_KMH", "30"))
SIM_SEED = int(os.getenv("SIM_SEED", "42"))
USE_MAPS_FOR_ROUTES = os.getenv("USE_MAPS_FOR_ROUTES", "true").strip().lower() == "true"

_DEFAULT_TRUSTED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
}


def trusted_hosts():
    """Hosts allowed by the Host-header middleware."""
    hosts = set(_DEFAULT_TRUSTED_HOSTS)
    for host in os.getenv("TRUSTED_HOSTS", "").split(","):
        host = host.strip().lower()
        if host:
            hosts.add(host)
    render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip().lower()
    if render_host:
        hosts.add(render_host)
    return hosts


def validate_required_config():
    if REQUIRE_MAPS_API_KEY and not GOOGLE_MAPS_API_KEY:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY (or MAPS_API_KEY) is required when "
            "REQUIRE_MAPS_API_KEY=true."
        )
