"""Simple walk-forward/backtest helper for retrain evaluation.

This script is intentionally minimal: it expects a CSV of features+target and
walks a rolling train/test window evaluating a pickled model callable.
"""
import argparse
import pandas as pd
import pickle
import os


def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def walk_forward(feature_csv: str, model_path: str, train_window: int = 500, test_window: int = 100):
    df = pd.read_csv(feature_csv)
    n = len(df)
    results = []
    m = load_model(model_path)
    for start in range(0, n - train_window - test_window + 1, test_window):
        train = df.iloc[start:start+train_window]
        test = df.iloc[start+train_window:start+train_window+test_window]
        if 'y' not in test.columns:
            break
        X_test = test.drop(columns=['y'])
        y_test = test['y']
        pred = m.predict(X_test.values if hasattr(X_test, 'values') else X_test)
        acc = (pred == y_test.values).mean()
        results.append({'start': start, 'acc': acc, 'n': len(y_test)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('feature_csv')
    parser.add_argument('model_path')
    parser.add_argument('--train', type=int, default=500)
    parser.add_argument('--test', type=int, default=100)
    args = parser.parse_args()
    res = walk_forward(args.feature_csv, args.model_path, args.train, args.test)
    for r in res:
        print(r)


if __name__ == '__main__':
    main()
