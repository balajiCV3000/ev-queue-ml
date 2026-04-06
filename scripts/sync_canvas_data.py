"""Embed results/_agg.json into the canvas DATA block."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "results" / "_agg.json"
CANVAS = Path.home() / ".cursor" / "projects" / "c-Users-cvbal-Desktop-ev-queue-3" / "canvases" / "ev-policy-comparison.canvas.tsx"

FIELDS = [
    "scenario", "policy", "average_wait_time", "average_travel_distance_km",
    "total_travel_distance_km", "avg_queue_length", "max_queue_length",
    "avg_station_utilization", "throughput_per_hour", "vehicles_served",
    "completion_rate", "abandoned_rate", "total_reward", "total_reward_std",
    "average_charging_completion_time",
]


def fmt_val(field, value):
    if field in ("scenario", "policy"):
        return f'"{value}"'
    return str(value)


def build_data_block(records):
    lines = ["const DATA: Row_[] = ["]
    for r in records:
        parts = [f"{f}: {fmt_val(f, r[f])}" for f in FIELDS]
        lines.append("  { " + ", ".join(parts) + " },")
    lines.append("];")
    return "\n".join(lines)


def main():
    records = json.loads(AGG.read_text(encoding="utf-8"))
    data_block = build_data_block(records)
    text = CANVAS.read_text(encoding="utf-8")
    text = re.sub(r"const DATA: Row_\[\] = \[.*?\];", data_block, text, count=1, flags=re.S)
    CANVAS.write_text(text, encoding="utf-8")
    print(f"Updated {CANVAS}")


if __name__ == "__main__":
    main()
