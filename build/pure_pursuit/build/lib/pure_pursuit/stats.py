import json
import os

STATS_FILE = "/workspace/ros2_ws/src/pure_pursuit/pure_pursuit/stats.json"

DEFAULT_STATS = {
    "runs": 0,
    "successes": 0,
    "total_position_error": 0.0,
    "total_orientation_error": 0.0
}


def load_stats():
    if not os.path.exists(STATS_FILE):
        return DEFAULT_STATS.copy()

    with open(STATS_FILE, "r") as f:
        return json.load(f)


def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)