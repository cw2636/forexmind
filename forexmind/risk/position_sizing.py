import math


def kelly_fraction(win_rate: float, payoff: float) -> float:
    """Compute Kelly fraction given win rate and payoff ratio.

    win_rate: probability of win (0..1)
    payoff: average win / average loss (positive)
    Returns fraction of bankroll to risk (0..1)."""
    if payoff <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = payoff
    p = win_rate
    q = 1 - p
    f = (b * p - q) / b
    return max(0.0, min(1.0, f))


def fractional_size(capital: float, risk_pct: float, stop_loss_pips: float, pip_value: float = 1.0) -> float:
    """Return position size (units) given capital and percent risk.

    - `risk_pct` in decimal (0.01 = 1%).
    - `stop_loss_pips` positive number of pips risked per unit.
    - `pip_value`: value per pip per unit (instrument-specific).
    """
    if stop_loss_pips <= 0:
        return 0.0
    risk_amount = capital * risk_pct
    units = risk_amount / (stop_loss_pips * pip_value)
    return max(0.0, units)


if __name__ == "__main__":
    print("Example Kelly (60% win, payoff 1.5):", kelly_fraction(0.6, 1.5))
    print("Example fractional size (10k capital, 1% risk, 50 pips):", fractional_size(10000, 0.01, 50, 1))
