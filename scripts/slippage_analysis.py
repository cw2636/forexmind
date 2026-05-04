"""Analyze TP/SL slippage and hit-rates from a trade fills file.

Input: CSV or JSONL where each record contains keys:
  instrument, side (buy/sell), expected_tp, expected_sl, exit_price, exit_reason, entry_price

Outputs simple summary per instrument and overall: TP hit rate, SL hit rate,
average slippage (in pips) when TP was expected, average slippage when SL.
"""
import argparse
import csv
import json
import math
import os
from typing import List, Dict


def is_jpy(instr: str) -> bool:
    return instr.endswith("JPY") or instr.endswith("jpy")


def to_pips(price_diff: float, instr: str) -> float:
    # common: 1 pip = 0.0001; JPY pairs 0.01
    pip = 0.01 if is_jpy(instr) else 0.0001
    return abs(price_diff) / pip


def load_rows(path: str) -> List[Dict]:
    if path.endswith('.jsonl') or path.endswith('.ndjson'):
        out = []
        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                out.append(json.loads(line))
        return out
    # assume CSV
    out = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(r)
    return out


def analyze(rows: List[Dict]) -> Dict:
    per_instr = {}
    totals = {"tp_hits": 0, "sl_hits": 0, "tp_expected": 0, "sl_expected": 0, "count": 0}
    for r in rows:
        instr = r.get('instrument') or r.get('symbol') or 'UNK'
        side = (r.get('side') or 'buy').lower()
        expected_tp = r.get('expected_tp') or r.get('expected_tp_price') or r.get('tp')
        expected_sl = r.get('expected_sl') or r.get('expected_sl_price') or r.get('sl')
        exit_price = r.get('exit_price') or r.get('fill_price') or r.get('exit')
        exit_reason = (r.get('exit_reason') or r.get('exitType') or '').lower()

        try:
            expected_tp = float(expected_tp) if expected_tp is not None and expected_tp != '' else None
            expected_sl = float(expected_sl) if expected_sl is not None and expected_sl != '' else None
            exit_price = float(exit_price) if exit_price is not None and exit_price != '' else None
        except Exception:
            continue

        rec = per_instr.setdefault(instr, {"tp_hits": 0, "sl_hits": 0, "tp_expected": 0, "sl_expected": 0,
                                            "tp_slippage_pips": [], "sl_slippage_pips": [], "count": 0})
        rec["count"] += 1
        totals["count"] += 1

        if expected_tp is not None:
            rec["tp_expected"] += 1
            totals["tp_expected"] += 1
            # did tp occur?
            tp_hit = False
            if exit_price is not None:
                if side == 'buy' and exit_price >= expected_tp:
                    tp_hit = True
                if side == 'sell' and exit_price <= expected_tp:
                    tp_hit = True
            if tp_hit:
                rec["tp_hits"] += 1
                totals["tp_hits"] += 1
            # slippage relative to expected_tp
            if exit_price is not None:
                slippage = to_pips(exit_price - expected_tp, instr)
                rec["tp_slippage_pips"].append(slippage)

        if expected_sl is not None:
            rec["sl_expected"] += 1
            totals["sl_expected"] += 1
            sl_hit = False
            if exit_price is not None:
                if side == 'buy' and exit_price <= expected_sl:
                    sl_hit = True
                if side == 'sell' and exit_price >= expected_sl:
                    sl_hit = True
            if sl_hit:
                rec["sl_hits"] += 1
                totals["sl_hits"] += 1
            if exit_price is not None:
                slippage = to_pips(exit_price - expected_sl, instr)
                rec["sl_slippage_pips"].append(slippage)

    # aggregate stats
    summary = {"per_instrument": {}, "totals": totals}
    for instr, rec in per_instr.items():
        tp_rate = rec["tp_hits"] / rec["tp_expected"] if rec["tp_expected"] else None
        sl_rate = rec["sl_hits"] / rec["sl_expected"] if rec["sl_expected"] else None
        tp_slip_avg = sum(rec["tp_slippage_pips"]) / len(rec["tp_slippage_pips"]) if rec["tp_slippage_pips"] else None
        sl_slip_avg = sum(rec["sl_slippage_pips"]) / len(rec["sl_slippage_pips"]) if rec["sl_slippage_pips"] else None
        summary["per_instrument"][instr] = {
            "count": rec["count"],
            "tp_expected": rec["tp_expected"],
            "tp_hits": rec["tp_hits"],
            "tp_hit_rate": tp_rate,
            "avg_tp_slippage_pips": tp_slip_avg,
            "sl_expected": rec["sl_expected"],
            "sl_hits": rec["sl_hits"],
            "sl_hit_rate": sl_rate,
            "avg_sl_slippage_pips": sl_slip_avg,
        }

    # totals rates
    totals["tp_hit_rate"] = totals["tp_hits"] / totals["tp_expected"] if totals["tp_expected"] else None
    totals["sl_hit_rate"] = totals["sl_hits"] / totals["sl_expected"] if totals["sl_expected"] else None
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('fills_path', help='CSV or JSONL of executed trades')
    p.add_argument('--out', help='JSON output path', default='artifacts/slippage_summary.json')
    args = p.parse_args()
    rows = load_rows(args.fills_path)
    summary = analyze(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
