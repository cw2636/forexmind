# ForexMind — Advanced Python Practice

> **Your personal interview prep lab.**  
> Every exercise uses real code from the ForexMind trading system so you're
> learning advanced Python *in context*, not in isolation.

---

## How Each Topic Works

Each folder contains three files:

| File | Purpose |
|------|---------|
| `lesson.md` | Core theory, interview Q&A, annotated examples from ForexMind |
| `exercises.py` | Scaffolded coding challenges — fill in the `TODO` sections |
| `solutions.py` | Complete working solutions with detailed comments |

**Workflow:**
1. Read `lesson.md` from top to bottom
2. Open `exercises.py` and attempt each exercise without looking at solutions
3. Run it: `python exercises.py`
4. Check `solutions.py` only after you've made a genuine attempt

---

## Topics

| # | Folder | Concept |
|---|--------|---------|
| 01 | `01_abc_protocols/` | Abstract Base Classes & Protocols |
| 02 | `02_dataclasses/` | Dataclasses — `__post_init__`, `field()`, frozen, `__slots__` |
| 03 | `03_descriptors/` | Descriptors — `__get__`, `__set__`, `__set_name__` |
| 04 | `04_decorators/` | Decorators — stacking, parametrized, class-based, `functools.wraps` |
| 05 | `05_context_managers/` | Context Managers — `__enter__`/`__exit__`, `asynccontextmanager` |
| 06 | `06_generators/` | Generators & Iterators — `yield`, `send()`, async generators |
| 07 | `07_type_system/` | Type System — `TypedDict`, `Protocol`, `Generic[T]`, `Annotated` |
| 08 | `08_asyncio/` | asyncio & Concurrency — `gather`, `TaskGroup`, `Lock`, `Queue` |
| 09 | `09_metaclasses/` | Metaclasses & `__init_subclass__` — plugin registry pattern |
| 10 | `10_design_patterns/` | Design Patterns — Strategy, Observer, Factory |
| 11 | `11_memory_performance/` | Memory & Performance — `__slots__`, `lru_cache`, profiling |
| 12 | `12_pattern_matching/` | Structural Pattern Matching — `match`/`case`, guards |
| 13 | `13_testing/` | Testing Mastery — `parametrize`, `mock`, fixtures |
| 14 | `14_closures/` | Closures, `nonlocal`, `__closure__`, free variables |

---

## Running the Exercises

```bash
# Activate the environment
source /home/wilson/ml-workspace/mlenv/bin/activate
cd /home/wilson/Forex/practice

# Run a single exercise file
python 01_abc_protocols/exercises.py

# Run all solutions to verify they work
python 01_abc_protocols/solutions.py
```

---

## Interview Tip

For every concept, be able to answer:
1. **What** is it? (one sentence definition)
2. **Why** does Python have it? (the problem it solves)
3. **When** do you use it vs. the alternatives?
4. **Show me** — write code on the spot

Each `lesson.md` is structured around exactly these four questions.
