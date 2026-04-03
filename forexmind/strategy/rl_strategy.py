"""
ForexMind — Reinforcement Learning Strategy (PPO Agent)
=========================================================
Uses Stable-Baselines3 PPO to train an agent that learns optimal
entry/exit timing by simulating trades in a Gymnasium environment.

The environment:
  - State:  last N indicator snapshots (flattened vector)
  - Action: 0=HOLD, 1=BUY, 2=SELL
  - Reward: P&L in pips (penalised for overtrading and losing trades)

Why RL?  Unlike supervised ML, RL directly optimises for profit rather
than directional accuracy — it learns *when* not to trade (very valuable).

Advanced Python:
  - gymnasium.Env subclass with proper observation/action spaces
  - numpy structured types for efficient state representation
  - Abstract method override typing
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, SupportsFloat

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from forexmind.strategy.base import BaseStrategy, StrategySignal
from forexmind.strategy.feature_engineering import build_feature_matrix, get_feature_columns
from forexmind.utils.helpers import atr_stop_loss, atr_take_profit, pip_size
from forexmind.utils.logger import get_logger
from forexmind.config.settings import get_settings

log = get_logger(__name__)

ACTION_HOLD = 0
ACTION_BUY  = 1
ACTION_SELL = 2


# ── Gymnasium Environment ─────────────────────────────────────────────────────

class ForexTradingEnv(gym.Env):
    """
    A forex trading environment for RL training.

    State:  Flattened feature vector of the last `window` bars.
    Action: Discrete(3) → HOLD / BUY / SELL
    Reward: Realised pip P&L when a position closes, minus penalties.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        window: int = 20,
        instrument: str = "EUR_USD",
        spread_pips: float = 1.5,
    ) -> None:
        super().__init__()
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium not installed. Run: pip install gymnasium")

        self._df = df
        self._feature_cols = feature_cols
        self._window = window
        self._instrument = instrument
        self._pip = pip_size(instrument)
        self._spread = spread_pips * self._pip
        self._n = len(df)

        # Observation: flat vector of last `window` x `n_features`
        n_features = len(feature_cols)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(window * n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)   # 0=HOLD, 1=BUY, 2=SELL

        # State tracking
        self._idx = window
        self._position: int = 0        # 0=flat, 1=long, -1=short
        self._entry_price: float = 0.0
        self._total_pips: float = 0.0
        self._trade_count: int = 0
        self._steps: int = 0

    def _get_obs(self) -> np.ndarray:
        """Return the flattened feature window, sanitized for NaN/inf."""
        window_df = self._df[self._feature_cols].iloc[self._idx - self._window:self._idx]
        obs = window_df.values.astype(np.float32).flatten()
        # Replace NaN/inf with 0 to prevent logits from becoming invalid
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._idx = self._window
        self._position = 0
        self._entry_price = 0.0
        self._total_pips = 0.0
        self._trade_count = 0
        self._steps = 0
        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        current_price = float(self._df["close"].iloc[self._idx])
        reward = 0.0
        info: dict[str, Any] = {}

        if action == ACTION_BUY and self._position == 0:
            # Open long
            self._position = 1
            self._entry_price = current_price + self._spread
            self._trade_count += 1

        elif action == ACTION_SELL and self._position == 0:
            # Open short
            self._position = -1
            self._entry_price = current_price - self._spread
            self._trade_count += 1

        elif action == ACTION_HOLD and self._position != 0:
            # Hold and receive ongoing P&L (mark-to-market, minor feedback)
            unrealised = self._position * (current_price - self._entry_price) / self._pip
            reward = unrealised * 0.001   # Small reward to guide agent

        elif self._position != 0 and (
            (self._position == 1 and action == ACTION_SELL) or
            (self._position == -1 and action == ACTION_BUY)
        ):
            # Close position
            exit_price = current_price - (self._spread if self._position == 1 else -self._spread)
            pips = self._position * (exit_price - self._entry_price) / self._pip
            self._total_pips += pips
            reward = pips
            self._position = 0
            self._entry_price = 0.0
            info["closed_pips"] = pips

        # Penalty for overtrading (more than 20 trades per episode is expensive)
        if self._trade_count > 20:
            reward -= 0.1

        self._idx += 1
        self._steps += 1

        terminated = self._idx >= self._n - 1
        # If still in position at end of episode, force close
        if terminated and self._position != 0:
            final_price = float(self._df["close"].iloc[-1])
            pips = self._position * (final_price - self._entry_price) / self._pip
            reward += pips
            self._total_pips += pips

        info["total_pips"] = self._total_pips
        info["position"] = self._position
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, False, info  # obs, reward, terminated, truncated, info

    def render(self) -> None:
        log.debug(f"RL Env: step={self._steps} pos={self._position} total_pips={self._total_pips:.1f}")


# ── RL Strategy ───────────────────────────────────────────────────────────────

class RLStrategy(BaseStrategy):
    """
    Reinforcement Learning (PPO) strategy.
    Uses the ForexTradingEnv with stable-baselines3 PPO agent.
    """

    name: ClassVar[str] = "rl_agent"

    def __init__(self, model_path: Path | None = None) -> None:
        self._ppo: "PPO | None" = None
        self._feature_cols: list[str] = []
        self._window = 20
        self._cfg = get_settings()
        model_path = model_path or (self._cfg.app.models_dir / "ppo_forex.zip")
        if model_path.exists():
            self._load(model_path)
        else:
            log.info(f"RL model not found at {model_path}. Run train() first.")

    def _load(self, path: Path) -> None:
        if not SB3_AVAILABLE:
            log.warning("stable-baselines3 not installed — RL strategy disabled")
            return
        try:
            self._ppo = PPO.load(str(path))
            # Feature cols saved alongside the model
            meta_path = path.parent / "ppo_meta.npy"
            if meta_path.exists():
                self._feature_cols = np.load(meta_path, allow_pickle=True).tolist()
            log.info(f"PPO model loaded from {path}")
        except Exception as e:
            log.error(f"Failed to load PPO model: {e}")

    def save(self, path: Path | None = None) -> None:
        if self._ppo is None:
            return
        path = path or (get_settings().app.models_dir / "ppo_forex.zip")
        self._ppo.save(str(path))
        meta_path = path.parent / "ppo_meta.npy"
        np.save(str(meta_path), np.array(self._feature_cols, dtype=object))
        log.info(f"PPO model saved to {path}")

    def train(
        self,
        df: pd.DataFrame,
        instrument: str = "EUR_USD",
        total_timesteps: int = 500_000,
    ) -> dict:
        if not (GYMNASIUM_AVAILABLE and SB3_AVAILABLE):
            raise ImportError("gymnasium and stable-baselines3 required for RL training")

        # Skip feature engineering if already pre-computed (multi-pair training path)
        if any(c in df.columns for c in ("rsi", "macd", "adx")):
            feat_df = df.replace([float("inf"), float("-inf")], float("nan")).dropna(
                subset=["rsi", "macd", "adx"]
            ).copy()
        else:
            feat_df = build_feature_matrix(df, add_target=False)
        self._feature_cols = get_feature_columns(feat_df)
        # Normalise features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feat_df[self._feature_cols] = scaler.fit_transform(feat_df[self._feature_cols].values)

        env = DummyVecEnv([lambda: ForexTradingEnv(
            feat_df, self._feature_cols, window=self._window, instrument=instrument
        )])

        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device="cpu",   # MlpPolicy trains faster on CPU
            verbose=0,
        )

        log.info(f"Training PPO for {total_timesteps:,} timesteps...")
        model.learn(total_timesteps=total_timesteps)
        self._ppo = model
        self.save()
        log.info("PPO training complete")
        return {"total_timesteps": total_timesteps}

    def generate_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
    ) -> StrategySignal:
        if self._ppo is None or not self._feature_cols:
            return self._hold_signal(instrument, timeframe, current_price, "RL model not trained")

        feat_df = build_feature_matrix(df, add_target=False)
        if len(feat_df) < self._window:
            return self._hold_signal(instrument, timeframe, current_price, "Not enough data for RL window")

        # Build observation vector
        window_data = feat_df[self._feature_cols].iloc[-self._window:].values.astype(np.float32)
        obs = window_data.flatten()

        action, _ = self._ppo.predict(obs, deterministic=True)
        action = int(action)

        if action == ACTION_HOLD:
            return self._hold_signal(instrument, timeframe, current_price, "RL agent: HOLD")

        direction = "BUY" if action == ACTION_BUY else "SELL"
        atr = float(feat_df["atr"].iloc[-1]) if "atr" in feat_df.columns else current_price * 0.001
        stop_loss = atr_stop_loss(current_price, atr, direction)
        take_profit = atr_take_profit(current_price, stop_loss, direction)

        return StrategySignal(
            instrument=instrument, timeframe=timeframe,
            direction=direction, confidence=0.65,   # RL doesn't give probabilities
            entry_price=current_price, stop_loss=stop_loss, take_profit=take_profit,
            reasoning=f"PPO RL agent action={action} ({direction})",
            source=self.name,
        )
