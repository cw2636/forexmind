"""
ForexMind — ML Strategy (LightGBM + LSTM)
===========================================
Two complementary ML models:

1. **LightGBM classifier** — gradient-boosted trees on 100+ tabular features.
   Fast to train, explainable via feature importance, handles missing values.
   Output: probability distribution over [BUY=1, HOLD=0, SELL=-1]

2. **LSTM sequence model** — PyTorch recurrent network that reads the last
   N bars as a time sequence to capture temporal patterns.
   Output: softmax probabilities over [DOWN, FLAT, UP]

Both are trained offline (train_models()) and loaded for inference.

Advanced Python concepts:
  - PyTorch nn.Module with forward()
  - sklearn Pipeline for preprocessing
  - joblib for model persistence
  - __slots__ on dataclasses for memory efficiency
  - torch.no_grad() context manager
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

# PyTorch LSTM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from forexmind.strategy.base import BaseStrategy, StrategySignal
from forexmind.strategy.feature_engineering import build_feature_matrix, get_feature_columns
from forexmind.utils.helpers import atr_stop_loss, atr_take_profit
from forexmind.utils.logger import get_logger
from forexmind.config.settings import get_settings

log = get_logger(__name__)

# Optional imports (soft-dependency)
try:
    import lightgbm as lgb
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("LightGBM or scikit-learn not installed — ML classifiers disabled")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── LSTM Architecture ─────────────────────────────────────────────────────────

class ForexLSTM(nn.Module):
    """
    Bidirectional LSTM with self-attention for forex direction prediction.

    Architecture:
      Input      → [batch, seq_len, features]
      LayerNorm  → stabilises input (key fix for NaN loss)
      BiLSTM     → [batch, seq_len, hidden*2]
      Attention  → weighted sum over time steps (focus on key patterns)
      Dropout    → regularisation
      FC         → [batch, 3]  (SELL / HOLD / BUY logits)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_size)   # stabilise inputs
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # Self-attention: learn which timesteps matter most
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)                       # [batch, seq, features]
        lstm_out, _ = self.lstm(x)                   # [batch, seq, hidden*2]
        # Attention weights over time steps
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)  # [batch, seq, 1]
        context = (attn_w * lstm_out).sum(dim=1)     # [batch, hidden*2]
        context = self.dropout(context)
        return self.fc(context)                      # [batch, 3]


# ── Lazy sequence dataset (avoids materialising 30+ GiB tensor) ───────────────

class SequenceDataset(Dataset):
    """
    Creates overlapping windows of length `seq_len` on-the-fly from a flat
    numpy array. Memory usage is O(N×F) not O(N×seq_len×F).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx: idx + self.seq_len], dtype=torch.float32)
        label = torch.tensor(self.y[idx + self.seq_len], dtype=torch.long)
        return x, label


# ── LightGBM strategy ─────────────────────────────────────────────────────────

class LightGBMStrategy(BaseStrategy):
    """
    Gradient-boosted tree classifier for forex direction.
    Trained on tabular features from FeatureEngineering pipeline.
    """

    name: ClassVar[str] = "lightgbm"

    def __init__(self, model_path: Path | None = None) -> None:
        self._model: Pipeline | None = None
        self._feature_cols: list[str] = []
        self._cfg = get_settings()
        model_path = model_path or (self._cfg.app.models_dir / "lgbm_forex.pkl")
        if model_path.exists():
            self._load(model_path)
        else:
            log.info(f"LightGBM model not found at {model_path}. Run train() first.")

    def _load(self, path: Path) -> None:
        if not SKLEARN_AVAILABLE:
            return
        try:
            state = joblib.load(path)
            self._model = state["model"]
            self._feature_cols = state["feature_cols"]
            log.info(f"LightGBM model loaded from {path}")
        except Exception as e:
            log.error(f"Failed to load LightGBM model: {e}")

    def save(self, path: Path | None = None) -> None:
        if not SKLEARN_AVAILABLE or self._model is None:
            return
        path = path or (get_settings().app.models_dir / "lgbm_forex.pkl")
        joblib.dump({"model": self._model, "feature_cols": self._feature_cols}, path)
        log.info(f"LightGBM model saved to {path}")

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the LightGBM classifier on a feature-engineered DataFrame.
        Uses TimeSeriesSplit (no look-ahead bias).

        Returns metrics dict: {accuracy, f1, feature_importances}
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("lightgbm and scikit-learn required for training")

        from sklearn.metrics import classification_report
        from sklearn.utils.class_weight import compute_class_weight

        # Skip feature engineering if already pre-computed (multi-pair training path)
        if "target" in df.columns:
            feat_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["rsi", "macd", "adx"]).copy()
        else:
            feat_df = build_feature_matrix(df, add_target=True)
        self._feature_cols = get_feature_columns(feat_df)

        X = feat_df[self._feature_cols].values
        y = feat_df["target"].values         # -1, 0, 1
        y_mapped = y + 1                     # LightGBM needs 0, 1, 2

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self._cfg.ml_config.get("cv_folds", 5))

        best_val_acc = 0.0
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_mapped[train_idx], y_mapped[val_idx]

            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            w_map = dict(zip(classes.tolist(), weights.tolist()))
            sample_weight = np.array([w_map.get(yi, 1.0) for yi in y_train])

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", lgb.LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=63,
                    max_depth=7,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )),
            ])
            pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight)

            val_acc = pipeline.score(X_val, y_val)
            log.debug(f"LightGBM fold {fold+1} val accuracy: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = pipeline

        self._model = best_model
        self.save()

        # Final report on last held-out fold
        y_pred = best_model.predict(X_val)
        report = classification_report(y_val, y_pred, target_names=["SELL", "HOLD", "BUY"], output_dict=True)
        log.info(f"LightGBM training complete. Best val accuracy: {best_val_acc:.4f}")

        feat_imp = best_model.named_steps["clf"].feature_importances_
        top_feats = sorted(
            zip(self._feature_cols, feat_imp),
            key=lambda x: x[1], reverse=True
        )[:10]
        log.info(f"Top 10 features: {top_feats}")

        return {"accuracy": best_val_acc, "report": report, "top_features": top_feats}

    def generate_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
    ) -> StrategySignal:
        if self._model is None or not self._feature_cols:
            return self._hold_signal(instrument, timeframe, current_price, "LightGBM model not trained")

        feat_df = build_feature_matrix(df, add_target=False)
        if feat_df.empty or not all(c in feat_df.columns for c in self._feature_cols):
            return self._hold_signal(instrument, timeframe, current_price, "Feature extraction failed")

        X = feat_df[self._feature_cols].iloc[[-1]].values   # Last row only
        proba = self._model.predict_proba(X)[0]             # [P(sell), P(hold), P(buy)]

        buy_prob = proba[2]
        sell_prob = proba[0]
        hold_prob = proba[1]
        confidence_threshold = self._cfg.ml_config.get("min_confidence_threshold", 0.55)

        if buy_prob > confidence_threshold and buy_prob > sell_prob:
            direction = "BUY"
            confidence = buy_prob
        elif sell_prob > confidence_threshold and sell_prob > buy_prob:
            direction = "SELL"
            confidence = sell_prob
        else:
            return self._hold_signal(
                instrument, timeframe, current_price,
                f"LightGBM confidence too low: buy={buy_prob:.2f} sell={sell_prob:.2f}"
            )

        atr = float(feat_df["atr"].iloc[-1]) if "atr" in feat_df.columns else current_price * 0.001
        stop_loss = atr_stop_loss(current_price, atr, direction)
        take_profit = atr_take_profit(current_price, stop_loss, direction)

        return StrategySignal(
            instrument=instrument, timeframe=timeframe,
            direction=direction, confidence=round(confidence, 4),
            entry_price=current_price, stop_loss=stop_loss, take_profit=take_profit,
            reasoning=(
                f"LightGBM: BUY={buy_prob:.3f} HOLD={hold_prob:.3f} SELL={sell_prob:.3f}"
            ),
            source=self.name,
        )


# ── LSTM Strategy ──────────────────────────────────────────────────────────────

class LSTMStrategy(BaseStrategy):
    """
    PyTorch Bidirectional LSTM strategy.
    Trained on sequences of length `seq_len` bars.
    """

    name: ClassVar[str] = "lstm"

    def __init__(self, seq_len: int = 60, model_path: Path | None = None) -> None:
        self._seq_len = seq_len
        self._model: ForexLSTM | None = None
        self._scaler: "StandardScaler | None" = None  # type: ignore[name-defined]
        self._feature_cols: list[str] = []
        self._best_params: dict = {}
        self._cfg = get_settings()
        model_path = model_path or (self._cfg.app.models_dir / "lstm_forex.pt")
        if model_path.exists():
            self._load(model_path)
        else:
            log.info(f"LSTM model not found at {model_path}. Run train() first.")

    def _load(self, path: Path) -> None:
        try:
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            self._feature_cols = checkpoint["feature_cols"]
            self._scaler = checkpoint["scaler"]
            self._seq_len = checkpoint.get("seq_len", self._seq_len)
            self._best_params = checkpoint.get("best_params", {})
            model = ForexLSTM(
                input_size=len(self._feature_cols),
                hidden_size=checkpoint.get("hidden_size", 128),
                num_layers=checkpoint.get("num_layers", 2),
                num_classes=checkpoint.get("num_classes", 2),
            ).to(DEVICE)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self._model = model
            log.info(f"LSTM model loaded from {path}")
        except Exception as e:
            log.error(f"Failed to load LSTM model: {e}")

    def save(self, path: Path | None = None) -> None:
        if self._model is None:
            return
        path = path or (get_settings().app.models_dir / "lstm_forex.pt")
        torch.save({
            "model_state": self._model.state_dict(),
            "feature_cols": self._feature_cols,
            "scaler": self._scaler,
            "hidden_size": self._model.hidden_size,
            "num_layers": self._model.num_layers,
            "num_classes": self._model.fc[-1].out_features,
            "seq_len": self._seq_len,
            "best_params": self._best_params,
        }, path)
        log.info(f"LSTM model saved to {path}")

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 1024,
        lr: float = 3e-4,
        target_accuracy: float = 0.70,
        max_rows: int = 200_000,
        warm_start: bool = False,
    ) -> dict:
        """
        Train BiLSTM with attention on the dataset.
        Runs a hyperparameter grid search over key params until target_accuracy is hit.

        Args:
            max_rows: Cap dataset to this many rows (last N kept — most recent is most relevant).
                      Keeps each epoch fast (<5 min) and avoids OOM.
            warm_start: If True, skip hyperparameter search and fine-tune the current
                        best model (used for progressive multi-pair training).
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for LSTM scaler")

        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.class_weight import compute_class_weight

        # Skip feature engineering if already pre-computed (multi-pair training path)
        if "target" in df.columns:
            feat_df = df.replace([np.inf, -np.inf], np.nan).copy()
        else:
            feat_df = build_feature_matrix(df, add_target=True)
        self._feature_cols = get_feature_columns(feat_df)

        # Sanitize features — drop inf/nan rows
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        feat_df = feat_df.dropna(subset=self._feature_cols)

        # Cap rows — keep the most recent data (temporally ordered)
        if len(feat_df) > max_rows:
            feat_df = feat_df.iloc[-max_rows:].copy()
            log.info(f"Dataset capped to last {max_rows:,} rows for LSTM training")

        # Binary classification: UP vs DOWN only — drop HOLD rows.
        # Baseline = 50%, target 58-64%. Cleaner signal than 3-class on M5/H1.
        feat_df = feat_df[feat_df["target"] != 0].copy()
        log.info(f"Binary classification: {len(feat_df):,} directional rows (HOLD dropped)")

        X_raw = feat_df[self._feature_cols].values.astype(np.float32)
        # Remap: -1 → 0 (SELL), +1 → 1 (BUY)
        y_raw = ((feat_df["target"].values + 1) // 2).astype(np.int64)

        split = int(len(X_raw) * 0.8)
        scaler = StandardScaler()
        X_scaled = X_raw.copy()
        X_scaled[:split] = scaler.fit_transform(X_raw[:split])
        X_scaled[split:] = scaler.transform(X_raw[split:])
        self._scaler = scaler

        # Clip extreme values after scaling
        X_scaled = np.clip(X_scaled, -5.0, 5.0)

        # Compute class weights for any remaining imbalance
        classes = np.unique(y_raw)
        weights = compute_class_weight("balanced", classes=classes, y=y_raw)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        NUM_CLASSES = 2

        # Hyperparameter grid — CPU-friendly sizes, smallest→largest so fast trials run first.
        # Binary classification → simpler task, smaller models work fine.
        PARAM_GRID = [
            {"hidden_size": 64,  "num_layers": 1, "seq_len": 30, "dropout": 0.2, "lr": 3e-4},
            {"hidden_size": 64,  "num_layers": 2, "seq_len": 30, "dropout": 0.2, "lr": 3e-4},
            {"hidden_size": 96,  "num_layers": 1, "seq_len": 48, "dropout": 0.25, "lr": 2e-4},
            {"hidden_size": 96,  "num_layers": 2, "seq_len": 30, "dropout": 0.25, "lr": 2e-4},
            {"hidden_size": 96,  "num_layers": 2, "seq_len": 48, "dropout": 0.25, "lr": 2e-4},
            {"hidden_size": 128, "num_layers": 2, "seq_len": 48, "dropout": 0.3, "lr": 2e-4},
            {"hidden_size": 128, "num_layers": 2, "seq_len": 48, "dropout": 0.3, "lr": 1e-4},
        ]

        # warm_start: reuse existing model + best params, just fine-tune (fewer epochs)
        if warm_start and self._model is not None and self._best_params:
            PARAM_GRID = [self._best_params]
            epochs = max(10, epochs // 3)
            log.info(f"Warm-start fine-tuning with params={self._best_params}, epochs={epochs}")

        best_val_acc = 0.0
        best_model = None
        best_params: dict = {}

        for trial_idx, params in enumerate(PARAM_GRID):
            seq_len = params["seq_len"]
            log.info(f"LSTM trial {trial_idx+1}/{len(PARAM_GRID)}: {params}")

            # Lazy datasets — no materialisation of 30+ GiB sequence tensors
            split_idx = int(len(X_scaled) * 0.8)
            train_ds = SequenceDataset(X_scaled[:split_idx + seq_len], y_raw[:split_idx + seq_len], seq_len)
            val_ds   = SequenceDataset(X_scaled[split_idx:], y_raw[split_idx:], seq_len)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, persistent_workers=True)
            val_dl   = DataLoader(val_ds, batch_size=batch_size * 2,
                                  num_workers=2, pin_memory=True, persistent_workers=True)

            model = ForexLSTM(
                input_size=len(self._feature_cols),
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                num_classes=NUM_CLASSES,
            ).to(DEVICE)

            current_lr = params["lr"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6
            )

            trial_best_acc = 0.0
            nan_strikes = 0
            no_improve = 0       # early-stopping counter
            ES_PATIENCE = 5      # stop if val_acc doesn't improve for 5 epochs

            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                nan_batch = False

                for Xb, yb in train_dl:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    logits = model(Xb)
                    loss = criterion(logits, yb)

                    if torch.isnan(loss):
                        nan_batch = True
                        break

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    total_loss += loss.item()

                if nan_batch:
                    nan_strikes += 1
                    current_lr *= 0.3
                    for pg in optimizer.param_groups:
                        pg["lr"] = current_lr
                    log.warning(f"Trial {trial_idx+1} epoch {epoch}: NaN loss — reducing LR to {current_lr:.2e}")
                    if nan_strikes >= 3:
                        log.warning(f"Trial {trial_idx+1}: too many NaN strikes, skipping")
                        break
                    continue

                # Validation
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for Xb, yb in val_dl:
                        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                        preds = model(Xb).argmax(dim=1)
                        correct += (preds == yb).sum().item()
                        total += len(yb)

                val_acc = correct / total if total > 0 else 0.0
                scheduler.step(val_acc)

                if val_acc > trial_best_acc:
                    trial_best_acc = val_acc
                    no_improve = 0
                else:
                    no_improve += 1

                if epoch % 5 == 0:
                    log.info(
                        f"Trial {trial_idx+1} Epoch {epoch}/{epochs} "
                        f"loss={total_loss/len(train_dl):.4f} val_acc={val_acc:.4f} "
                        f"best={trial_best_acc:.4f}"
                    )

                if no_improve >= ES_PATIENCE:
                    log.info(f"Trial {trial_idx+1} early stop at epoch {epoch} (no improvement for {ES_PATIENCE} epochs)")
                    break

            log.info(f"Trial {trial_idx+1} best val_acc={trial_best_acc:.4f} params={params}")

            if trial_best_acc > best_val_acc:
                best_val_acc = trial_best_acc
                best_model = model
                best_params = params
                self._seq_len = seq_len
                self._best_params = params

            if best_val_acc >= target_accuracy:
                log.info(f"Target accuracy {target_accuracy:.0%} reached — stopping search")
                break

        self._model = best_model
        self.save()
        log.info(f"LSTM training complete. Best val accuracy: {best_val_acc:.4f} | params: {best_params}")
        return {"accuracy": best_val_acc, "best_params": best_params}


    def generate_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
    ) -> StrategySignal:
        if self._model is None:
            return self._hold_signal(instrument, timeframe, current_price, "LSTM not trained")

        feat_df = build_feature_matrix(df, add_target=False)
        if len(feat_df) < self._seq_len:
            return self._hold_signal(instrument, timeframe, current_price, "Not enough data for LSTM sequence")

        X_raw = feat_df[self._feature_cols].values[-self._seq_len:].astype(np.float32)
        if self._scaler is not None:
            X_raw = self._scaler.transform(X_raw)

        x = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(x)
            proba = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Binary model: proba has 2 elements [SELL, BUY]; 3-class: [SELL, HOLD, BUY]
        num_classes = self._model.fc[-1].out_features
        if num_classes == 2:
            sell_p, buy_p = float(proba[0]), float(proba[1])
            hold_p = 0.0
        else:
            sell_p, hold_p, buy_p = float(proba[0]), float(proba[1]), float(proba[2])

        threshold = self._cfg.ml_config.get("min_confidence_threshold", 0.55)

        if buy_p > threshold and buy_p > sell_p:
            direction, confidence = "BUY", buy_p
        elif sell_p > threshold and sell_p > buy_p:
            direction, confidence = "SELL", sell_p
        else:
            return self._hold_signal(
                instrument, timeframe, current_price,
                f"LSTM confidence low: BUY={buy_p:.2f} SELL={sell_p:.2f}"
            )

        atr = float(feat_df["atr"].iloc[-1]) if "atr" in feat_df.columns else current_price * 0.001
        stop_loss = atr_stop_loss(current_price, atr, direction)
        take_profit = atr_take_profit(current_price, stop_loss, direction)

        return StrategySignal(
            instrument=instrument, timeframe=timeframe,
            direction=direction, confidence=round(confidence, 4),
            entry_price=current_price, stop_loss=stop_loss, take_profit=take_profit,
            reasoning=f"LSTM: BUY={buy_p:.3f} SELL={sell_p:.3f}" + (f" HOLD={hold_p:.3f}" if hold_p else ""),
            source=self.name,
        )
