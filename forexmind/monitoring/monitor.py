import json
import os
from typing import Dict


class ModelMonitor:
    """Simple file-backed monitor for retrain metrics.

    It records metrics per model artifact and exposes a simple drift check.
    Works with the CI artifacts folder produced by the retrain workflow.
    """

    def __init__(self, history_path: str = "artifacts/metrics_history.json"):
        self.history_path = history_path
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)

    def record_metrics(self, model_name: str, metrics: Dict):
        data = self._load_history()
        entry = {"model": model_name, "metrics": metrics}
        data.append(entry)
        self._save_history(data)

    def latest(self):
        data = self._load_history()
        return data[-1] if data else None

    def baseline(self, lookback: int = 2):
        data = self._load_history()
        if len(data) < lookback:
            return None
        return data[-lookback]

    def check_accuracy_drop(self, baseline_metrics: Dict, new_metrics: Dict, drop_threshold: float = 0.03) -> bool:
        """Return True if new_metrics show accuracy drop greater than threshold.

        Expects metrics dict with an `accuracy` float in range [0,1].
        """
        try:
            base_acc = float(baseline_metrics.get("accuracy", 0))
            new_acc = float(new_metrics.get("accuracy", 0))
        except Exception:
            return False
        return (base_acc - new_acc) >= drop_threshold

    def _load_history(self):
        if not os.path.exists(self.history_path):
            return []
        with open(self.history_path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return []

    def _save_history(self, data):
        with open(self.history_path, "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    # quick CLI helper to print latest metrics
    m = ModelMonitor()
    latest = m.latest()
    if latest:
        print("Latest:", json.dumps(latest, indent=2))
    else:
        print("No metrics recorded yet.")
