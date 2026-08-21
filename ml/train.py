"""Train the LSTM RUL/fault model on physics-simulated tank scenarios and
export everything the GitHub-Pages dashboard needs:

    docs/model.json        LSTM weights (browser forward pass)
    docs/config.json       parts, thresholds, scaler, meta
    docs/live_stream.json  demo telemetry replay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .lstm import LSTMModel, predict_rul, train
from .parts import INPUT_FEATURES, PART_ORDER, RUL_CAP_STEPS, PARTS, PART_ORDER as _PO
from .scenarios import build_dataset, save_demo

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--demo-steps", type=int, default=2000)
    ap.add_argument("--w-reg", type=float, default=2.0)
    ap.add_argument("--w-cls", type=float, default=1.5,
                    help="classification loss weight")
    ap.add_argument("--quick", action="store_true",
                    help="small dataset for smoke testing")
    args = ap.parse_args()

    print("Generating physics-simulated scenarios ...")
    data = build_dataset(window_samples=256 if args.quick else 512,
                         sample_rate=500.0,
                         window=args.window, stride=args.stride,
                         demo_steps=args.demo_steps)
    if args.quick:
        data["per_scenario"] = data["per_scenario"][:4]

    n_classes = len(data["class_names"])
    model = LSTMModel(D=len(INPUT_FEATURES), H=args.hidden,
                      R=len(PART_ORDER), C=n_classes, seed=0)

    # Shuffle scenario order so the held-out split spans all fault types.
    rng = np.random.default_rng(11)
    order = rng.permutation(len(data["per_scenario"]))
    shuffled = [data["per_scenario"][i] for i in order]
    n_val = max(1, int(len(shuffled) * 0.18))
    val_sets = shuffled[-n_val:]
    train_sets = shuffled[:-n_val]
    print(f"scenarios: train={len(train_sets)} val={len(val_sets)} "
          f"windows: train={sum(len(s['X']) for s in train_sets)}")

    print("Training LSTM (RUL regression + fault classification) ...")
    train(model, train_sets, val_sets, epochs=args.epochs,
          w_reg=args.w_reg, w_cls=args.w_cls)

    # ---- evaluation on held-out scenarios --------------------------------
    print("\nEvaluation (held-out scenarios):")
    for s in val_sets:
        y_true = s["Y"]
        rul_pred = np.stack([predict_rul(model, s["X"][i])[0]
                             for i in range(len(s["X"]))])
        mae = np.mean(np.abs(rul_pred - y_true)) * RUL_CAP_STEPS
        per_part = np.mean(np.abs(rul_pred - y_true), axis=0) * RUL_CAP_STEPS
        cls_pred = np.argmax(np.stack([predict_rul(model, s["X"][i])[1]
                                       for i in range(len(s["X"]))]), axis=1)
        acc = float(np.mean(cls_pred == s["L"]))
        parts_str = ", ".join(f"{p}={v:.0f}" for p, v in zip(PART_ORDER, per_part))
        print(f"  {s['name']:<28} RUL MAE={mae:6.1f} steps  "
              f"({parts_str})  cls_acc={acc:.2f}")

    # ---- export ----------------------------------------------------------
    DOCS.mkdir(exist_ok=True)
    model.save(str(DOCS / "model.json"))
    save_demo(data["stream"], str(DOCS / "live_stream.json"))
    config = {
        "parts": PARTS,
        "part_order": PART_ORDER,
        "input_features": INPUT_FEATURES,
        "scaler": data["scaler"],
        "class_names": data["class_names"],
        "rul_cap_steps": RUL_CAP_STEPS,
        "fail_health": 25.0,
        "window": args.window,
        "dt": 0.05,
        "model": {"hidden": args.hidden},
    }
    with open(DOCS / "config.json", "w") as fh:
        json.dump(config, fh)
    print(f"\nExported docs/model.json, docs/config.json, "
          f"docs/live_stream.json ({len(data['stream']['records'])} steps)")


if __name__ == "__main__":
    main()