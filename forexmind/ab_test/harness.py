import importlib.util
import pickle
import os
import pandas as pd


def load_model(path):
    if path.endswith('.pkl') or path.endswith('.joblib'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    raise RuntimeError('Unsupported model format')


def simulate(models: dict, features: pd.DataFrame) -> dict:
    """Run each model against `features` and simulate simple returns.

    models: dict[name->callable predict function OR model object]
    features: DataFrame of features
    Returns dict of name->accuracy and pseudo-P&L (counts).
    """
    results = {}
    y_true = features.get('y')
    X = features.drop(columns=['y'], errors='ignore')
    for name, m in models.items():
        if hasattr(m, 'predict'):
            pred = m.predict(X.values if hasattr(X, 'values') else X)
        elif callable(m):
            pred = m(X)
        else:
            raise RuntimeError('Model not callable')
        acc = (pred == (y_true.values if y_true is not None else pred)).mean() if y_true is not None else None
        pnl = float((pred == 1).sum() - (pred == 0).sum())
        results[name] = {"accuracy": acc, "pseudo_pnl": pnl}
    return results


def run_from_files(model_paths: dict, feature_csv: str):
    df = pd.read_csv(feature_csv)
    models = {}
    for name, path in model_paths.items():
        models[name] = load_model(path)
    return simulate(models, df)


if __name__ == '__main__':
    print('This is a minimal A/B harness. Use run_from_files to compare models.')
