"""Pure-numpy LSTM for sequence-to-one regression (RUL) with an
auxiliary classification head (fault type), trained with Adam and
truncated BPTT.  Weights are exported to JSON so the identical forward
pass runs in the browser dashboard (docs/lstm.js).

Architecture
------------
    x_t (D) --[LSTM, hidden H]--> h_T --[sigmoid]--> RUL per part (R)
                                      |--[softmax]--> fault class (C)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def tanh(z):
    return np.tanh(z)


class LSTMModel:
    GATES = ("f", "i", "c", "o")

    def __init__(self, D: int, H: int, R: int, C: int, seed: int = 0):
        self.D, self.H, self.R, self.C = D, H, R, C
        rng = np.random.default_rng(seed)
        self.p = {}
        for g in self.GATES:
            self.p[f"W{g}"] = rng.normal(0, 0.08, (D, H))
            self.p[f"U{g}"] = rng.normal(0, 0.08, (H, H))
            self.p[f"b{g}"] = np.zeros(H)
        self.p["Wy"] = rng.normal(0, 0.08, (H, R))
        self.p["by"] = np.zeros(R)
        self.p["Wcls"] = rng.normal(0, 0.08, (H, C))
        self.p["bcls"] = np.zeros(C)

    # ------------------------------------------------------------------
    def forward(self, X: np.ndarray) -> dict:
        """X: (T, D).  Returns cache with h_final, reg out, cls out."""
        T = X.shape[0]
        h = np.zeros(self.H)
        c = np.zeros(self.H)
        hs, cs, gates = [], [], []
        for t in range(T):
            x = X[t]
            f = sigmoid(x @ self.p["Wf"] + h @ self.p["Uf"] + self.p["bf"])
            i = sigmoid(x @ self.p["Wi"] + h @ self.p["Ui"] + self.p["bi"])
            c_t = tanh(x @ self.p["Wc"] + h @ self.p["Uc"] + self.p["bc"])
            o = sigmoid(x @ self.p["Wo"] + h @ self.p["Uo"] + self.p["bo"])
            c = f * c + i * c_t
            h = o * tanh(c)
            hs.append(h.copy())
            cs.append(c.copy())
            gates.append({"f": f, "i": i, "c_t": c_t, "o": o})
        reg = sigmoid(h @ self.p["Wy"] + self.p["by"])
        logits = h @ self.p["Wcls"] + self.p["bcls"]
        exp = np.exp(logits - logits.max())
        cls = exp / exp.sum()
        return {"h_final": h, "reg": reg, "cls": cls,
                "hs": hs, "cs": cs, "gates": gates, "X": X}

    def loss(self, cache: dict, y_reg: np.ndarray, y_cls: int,
             w_reg: float = 2.0, w_cls: float = 0.4) -> float:
        mse = float(np.mean((cache["reg"] - y_reg) ** 2))
        ce = -np.log(np.clip(cache["cls"][y_cls], 1e-12, 1.0))
        return w_reg * mse + w_cls * ce

    # ------------------------------------------------------------------
    def backward(self, cache: dict, y_reg: np.ndarray, y_cls: int) -> dict:
        """BPTT from the final-step outputs (sequence-to-one)."""
        T = cache["X"].shape[0]
        g = {}
        for k in self.p:
            g[k] = np.zeros_like(self.p[k])

        d_reg = (cache["reg"] - y_reg) * cache["reg"] * (1 - cache["reg"])
        g["Wy"] += np.outer(cache["h_final"], d_reg)
        g["by"] += d_reg

        soft = cache["cls"]
        d_cls = soft.copy()
        d_cls[y_cls] -= 1.0
        g["Wcls"] += np.outer(cache["h_final"], d_cls)
        g["bcls"] += d_cls

        d_h = d_reg @ self.p["Wy"].T + d_cls @ self.p["Wcls"].T
        d_c = np.zeros(self.H)

        for t in range(T - 1, -1, -1):
            x = cache["X"][t]
            h_prev = cache["hs"][t - 1] if t > 0 else np.zeros(self.H)
            gate = cache["gates"][t]
            f, i, c_t, o = gate["f"], gate["i"], gate["c_t"], gate["o"]
            c_cur = cache["cs"][t]
            h_cur = cache["hs"][t]

            d_c += d_h * o * (1 - tanh(c_cur) ** 2)
            d_o = d_h * tanh(c_cur)
            d_i = d_c * c_t
            d_c_t = d_c * i
            d_f = d_c * cache["cs"][t - 1] if t > 0 else d_c * np.zeros(self.H)

            g["Uo"] += np.outer(h_prev, d_o * o * (1 - o))
            g["Wo"] += np.outer(x, d_o * o * (1 - o))
            g["bo"] += d_o * o * (1 - o)

            g["Uc"] += np.outer(h_prev, d_c_t * (1 - c_t ** 2))
            g["Wc_"] = np.outer(x, d_c_t * (1 - c_t ** 2))
            g["bc"] += d_c_t * (1 - c_t ** 2)

            g["Ui"] += np.outer(h_prev, d_i * i * (1 - i))
            g["Wi"] += np.outer(x, d_i * i * (1 - i))
            g["bi"] += d_i * i * (1 - i)

            g["Uf"] += np.outer(h_prev, d_f * f * (1 - f))
            g["Wf"] += np.outer(x, d_f * f * (1 - f))
            g["bf"] += d_f * f * (1 - f)

            # Merge Wc_ (cell-input matrix) into the Wc slot.
            g["Wc"] += g.pop("Wc_")
            g["bc"] += 0

            d_h = (d_o * o * (1 - o)) @ self.p["Uo"].T \
                + (d_c_t * (1 - c_t ** 2)) @ self.p["Uc"].T \
                + (d_i * i * (1 - i)) @ self.p["Ui"].T \
                + (d_f * f * (1 - f)) @ self.p["Uf"].T
            d_c = d_f * f
        return g

    # ------------------------------------------------------------------
    def clip_grads(self, g: dict, norm: float = 1.0) -> dict:
        total = np.sqrt(sum(float(np.sum(v ** 2)) for v in g.values()))
        if total > norm:
            scale = norm / total
            g = {k: v * scale for k, v in g.items()}
        return g

    def to_json(self) -> dict:
        def fmt(arr):
            if arr.ndim == 2:
                return [[round(x, 5) for x in row] for row in arr]
            return [round(x, 5) for x in arr]

        return {
            "D": self.D, "H": self.H, "R": self.R, "C": self.C,
            "params": {k: fmt(arr) for k, arr in self.p.items()},
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_json(), fh)


class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k in params:
            m = self.m.get(k, np.zeros_like(params[k]))
            v = self.v.get(k, np.zeros_like(params[k]))
            m = self.b1 * m + (1 - self.b1) * grads[k]
            v = self.b2 * v + (1 - self.b2) * grads[k] ** 2
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.m[k], self.v[k] = m, v


def train(model: LSTMModel, train_sets: list[dict], val_sets: list[dict],
          epochs: int = 25, batch: int = 64, lr: float = 0.001,
          grad_clip: float = 1.0, log_every: int = 200,
          w_reg: float = 2.0, w_cls: float = 0.4,
          weight_decay: float = 1e-3) -> dict:
    """train_sets/val_sets: lists of {'X': (n,T,D), 'Y': (n,R), 'L': (n,)}."""
    adam = Adam(lr=lr)
    xs = np.concatenate([s["X"] for s in train_sets])
    ys = np.concatenate([s["Y"] for s in train_sets])
    ls = np.concatenate([s["L"] for s in train_sets])
    n = len(xs)
    history = {"train": [], "val": []}
    best_val_mse = float("inf")
    best_params = None

    for epoch in range(epochs):
        effective_lr = lr * (0.92 ** epoch)
        adam.lr = effective_lr
        perm = np.random.default_rng(epoch).permutation(n)
        total_loss = 0.0
        steps = 0
        t0 = time.time()
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            g = None
            batch_loss = 0.0
            for i in idx:
                cache = model.forward(xs[i])
                batch_loss += model.loss(cache, ys[i], int(ls[i]),
                                         w_reg=w_reg, w_cls=w_cls)
                gi = model.backward(cache, ys[i], int(ls[i]))
                g = gi if g is None else {k: g[k] + gi[k] for k in g}
            model.clip_grads(g, grad_clip)
            adam.step(model.p, g)
            for k in model.p:
                model.p[k] -= effective_lr * weight_decay * model.p[k]
            total_loss += batch_loss
            steps += 1
            if steps % log_every == 0:
                print(f"  epoch {epoch + 1} batch {steps} loss "
                      f"{batch_loss / len(idx):.4f} ({time.time() - t0:.0f}s)")
                t0 = time.time()

        tr_loss = total_loss / n
        val_mse = evaluate_mse(model, val_sets)
        history["train"].append(tr_loss)
        history["val"].append(val_mse)
        print(f"epoch {epoch + 1}/{epochs}  train={tr_loss:.4f}  "
              f"val_RUL_MSE={val_mse:.4f}")
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = {k: v.copy() for k, v in model.p.items()}

    if best_params is not None:
        model.p = best_params
    return history


def evaluate_mse(model: LSTMModel, sets: list[dict]) -> float:
    total, n = 0.0, 0
    for s in sets:
        for i in range(len(s["X"])):
            cache = model.forward(s["X"][i])
            total += float(np.mean((cache["reg"] - s["Y"][i]) ** 2))
            n += 1
    return total / n


def evaluate_loss(model: LSTMModel, sets: list[dict],
                  w_reg: float = 2.0, w_cls: float = 0.4) -> float:
    total, n = 0.0, 0
    for s in sets:
        for i in range(len(s["X"])):
            cache = model.forward(s["X"][i])
            total += model.loss(cache, s["Y"][i], int(s["L"][i]),
                                w_reg=w_reg, w_cls=w_cls)
            n += 1
    return total / n


def predict_rul(model: LSTMModel, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (RUL[0..1] per part, fault-class probabilities)."""
    cache = model.forward(X)
    return cache["reg"], cache["cls"]