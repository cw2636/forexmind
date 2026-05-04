import os
import json
import shutil
from .monitor import ModelMonitor


def evaluate_and_rollback(artifacts_dir: str = "artifacts",
                          models_dir: str = "forexmind/models",
                          active_name: str = "active_model.pkl",
                          threshold_acc_drop: float = 0.03):
    """Evaluate latest metrics and rollback if accuracy drops beyond threshold.

    Behavior:
      - Reads `artifacts/retrain_metrics.json` if present and records it.
      - Compares to previous metrics recorded in `artifacts/metrics_history.json`.
      - If accuracy drop exceeds `threshold_acc_drop`, it will attempt to restore
        the previously active model by renaming files.
    """
    metrics_file = os.path.join(artifacts_dir, "retrain_metrics.json")
    monitor = ModelMonitor(history_path=os.path.join(artifacts_dir, "metrics_history.json"))

    if not os.path.exists(metrics_file):
        print("No retrain metrics found at", metrics_file)
        return False

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    model_name = metrics.get("model_name", "new_model")
    perf = metrics.get("metrics", {})

    baseline = monitor.baseline()
    monitor.record_metrics(model_name, perf)

    if baseline and monitor.check_accuracy_drop(baseline.get("metrics", {}), perf, drop_threshold=threshold_acc_drop):
        print("Detected accuracy drop — initiating rollback")
        # Attempt to find previous model file and restore it as active
        previous_models = sorted([p for p in os.listdir(models_dir) if p.endswith(('.pkl', '.pt'))])
        if not previous_models:
            print("No previous models found to rollback to")
            return False
        prev = previous_models[-2] if len(previous_models) >= 2 else previous_models[0]
        prev_path = os.path.join(models_dir, prev)
        active_path = os.path.join(models_dir, active_name)
        try:
            # backup current active
            if os.path.exists(active_path):
                shutil.copy2(active_path, active_path + ".bak")
            shutil.copy2(prev_path, active_path)
            print(f"Rolled back to {prev}")
            return True
        except Exception as e:
            print("Rollback failed:", e)
            return False

    print("No rollback needed — performance within expected range")
    return False


if __name__ == "__main__":
    evaluate_and_rollback()
