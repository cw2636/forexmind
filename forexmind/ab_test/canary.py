import json
import os
import random
import threading
from typing import Dict, List


class CanaryManager:
    """Simple canary manager to route a fraction of trades to variant models and record outcomes.

    Usage:
      cm = CanaryManager(config={"models": {"control":0.8, "variant":0.2}})
      model = cm.choose_model_for_trade(trade_id)
      # execute trade with chosen model
      cm.record_outcome(trade_id, model_name, outcome_dict)

    Records are appended to `forexmind/data/ab_live_results.jsonl` for later analysis.
    """

    def __init__(self, config: Dict = None, log_path: str = "forexmind/data/ab_live_results.jsonl"):
        self.config = config or {"models": {"control": 0.9, "variant": 0.1}}
        self.lock = threading.Lock()
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # normalize weights
        models = self.config.get("models", {})
        total = sum(models.values()) if models else 1.0
        if total <= 0:
            raise ValueError("Invalid model weights")
        self.models = [(k, v / total) for k, v in models.items()]

    def choose_model_for_trade(self, trade_id: str) -> str:
        r = random.random()
        cum = 0.0
        for name, weight in self.models:
            cum += weight
            if r <= cum:
                return name
        return self.models[-1][0]

    def record_outcome(self, trade_id: str, model_name: str, outcome: Dict):
        """Append an outcome record (JSON) to the log file.

        outcome should include keys like: instrument, side, entry_price,
        exit_price, exit_reason (tp/sl/close), pnl, timestamp.
        """
        rec = {"trade_id": trade_id, "model": model_name, "outcome": outcome}
        line = json.dumps(rec)
        with self.lock:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def read_results(self) -> List[Dict]:
        if not os.path.exists(self.log_path):
            return []
        out = []
        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out


if __name__ == "__main__":
    cm = CanaryManager()
    print("Models:", cm.models)
    # demo choose
    for i in range(5):
        print(i, cm.choose_model_for_trade(str(i)))
