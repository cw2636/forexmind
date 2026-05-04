# Topic 13 — Testing Mastery

## The Four Interview Questions

---

### 1. WHAT is it?

`pytest` is Python's de-facto testing framework. Beyond simple `assert` statements,
it provides fixtures (shared setup), parametrize (data-driven tests), mocking
(replacing real dependencies), and powerful plugin support.

---

### 2. WHY does Python have it?

Without testing, refactoring is dangerous. For a trading system processing real
money, untested code is a liability. Good tests:
- Catch regressions before they reach production
- Document expected behaviour
- Enable confident refactoring

---

### 3. Key Testing Concepts

| Concept | Purpose |
|---------|---------|
| `@pytest.fixture` | Reusable setup/teardown — avoids duplication |
| `@pytest.mark.parametrize` | Run one test with many input combinations |
| `MagicMock(spec=X)` | Replace real dependency with a controllable fake |
| `patch("module.Class")` | Temporarily replace class/function in a module |
| `AsyncMock` | Mock for `async` functions |
| `monkeypatch` | pytest's built-in patching fixture |
| `pytest.raises(E)` | Assert that a specific exception is raised |
| `pytest.approx` | Float comparison with tolerance |
| `capfd` / `capsys` | Capture stdout/stderr output |

---

### 4. SHOW ME

```python
import pytest
from unittest.mock import MagicMock, patch

# ── Fixture ───────────────────────────────────────────────────────────────────
@pytest.fixture
def engine():
    """Creates a fresh TradingEngine for each test that requests it."""
    client = MagicMock()
    client.get_price.return_value = 1.1050
    client.place_order.return_value = {"id": "T001"}
    return TradingEngine(client)

def test_engine_places_trade(engine):
    # No setup code needed — fixture provides the engine
    result = engine.process_signal(buy_signal)
    assert result["status"] == "placed"


# ── parametrize ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("pair,expected", [
    ("EUR_USD", 1.1050),
    ("GBP_USD", 1.2600),
    ("USD_JPY", 130.0),
])
def test_fetch_price(pair, expected):
    assert fetch_price(pair) == expected


# ── Mock — verify calls ───────────────────────────────────────────────────────
def test_api_called_once():
    mock_client = MagicMock(spec=OandaClient)
    mock_client.get_price.return_value = 1.1050
    engine = TradingEngine(mock_client)
    engine.process_signal(buy_signal)
    mock_client.get_price.assert_called_once_with("EUR_USD")


# ── patch() ───────────────────────────────────────────────────────────────────
def test_with_patch():
    with patch("mymodule.time.sleep") as mock_sleep:
        result = function_that_sleeps()
        mock_sleep.assert_called_once()

# As decorator
@patch("mymodule.OandaClient")
def test_decorated(MockClient):
    inst = MockClient.return_value
    inst.get_price.return_value = 1.1050
    ...


# ── pytest.raises ─────────────────────────────────────────────────────────────
def test_raises():
    with pytest.raises(ValueError, match="must be positive"):
        calculate_position_size(balance=-100, risk_pct=2.0)


# ── Fixture scope ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")   # created once per test session
def db_connection():
    conn = create_db()
    yield conn                     # yield + fixture = setup/teardown
    conn.close()                   # runs after all tests complete

@pytest.fixture(scope="module")    # once per test file
@pytest.fixture(scope="class")     # once per test class
@pytest.fixture                    # default: once per test function
```

---

## Common Interview Mistakes

1. **Not using `spec=` in MagicMock** — without `spec`, mock accepts ANY attribute name;
   typos in test code are silently swallowed
2. **Patching the wrong location** — patch where it's *used*, not where it's *defined*
3. **Fixtures without `yield` teardown** — if you need cleanup, use `yield` 
4. **`pytest.approx` for floats** — never use `==` for floats in tests
5. **Testing implementation, not behaviour** — good tests verify WHAT, not HOW
