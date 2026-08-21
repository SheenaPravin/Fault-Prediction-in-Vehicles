/* Pure-JS LSTM forward pass mirroring ml/lstm.py (LSTMModel.forward).
 * Loaded weights come from docs/model.json (trained offline in numpy). */

"use strict";

const LSTM = {
  _sigmoid(z) { return 1.0 / (1.0 + Math.exp(-Math.max(-40, Math.min(40, z)))); },
  _tanh(z) { const e = Math.exp(2 * z); return (e - 1) / (e + 1); },

  /* matVec: W is (D,H) row-major array-of-arrays, v is D-vector -> H-vector */
  _matVec(W, v) {
    const H = W[0].length, out = new Array(H).fill(0);
    for (let j = 0; j < H; j++) {
      let s = 0;
      for (let i = 0; i < v.length; i++) s += W[i][j] * v[i];
      out[j] = s;
    }
    return out;
  },
  _add(a, b) { return a.map((x, i) => x + b[i]); },
  _mul(a, b) { return a.map((x, i) => x * b[i]); },
  _sigArr(a) { return a.map(x => this._sigmoid(x)); },
  _tanhArr(a) { return a.map(x => this._tanh(x)); },

  softmax(z) {
    const m = Math.max(...z);
    const e = z.map(x => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(x => x / s);
  },

  /* forward(model, X): X = [ [f0..fD-1], ... ] length T.
   * Returns { reg: RUL[0..1] per part, cls: fault-class probabilities }. */
  forward(model, X) {
    const p = model.params, H = model.H, T = X.length;
    let h = new Array(H).fill(0), c = new Array(H).fill(0);
    for (let t = 0; t < T; t++) {
      const x = X[t];
      const f = this._sigArr(this._add(this._add(this._matVec(p.Wf, x), this._matVec(p.Uf, h)), p.bf));
      const i = this._sigArr(this._add(this._add(this._matVec(p.Wi, x), this._matVec(p.Ui, h)), p.bi));
      const ct = this._tanhArr(this._add(this._add(this._matVec(p.Wc, x), this._matVec(p.Uc, h)), p.bc));
      const o = this._sigArr(this._add(this._add(this._matVec(p.Wo, x), this._matVec(p.Uo, h)), p.bo));
      c = this._add(this._mul(f, c), this._mul(i, ct));
      h = this._mul(o, this._tanhArr(c));
    }
    const reg = this._sigArr(this._add(this._matVec(p.Wy, h), p.by));
    const cls = this.softmax(this._add(this._matVec(p.Wcls, h), p.bcls));
    return { reg, cls };
  }
};

/* window is the rolling sequence buffer (T x D) */
function rollingWindow(size, dim, fill) {
  const buf = [];
  for (let i = 0; i < size; i++) buf.push(fill.slice());
  return {
    push(vec) { buf.push(vec); if (buf.length > size) buf.shift(); return buf.slice(); },
    current: () => buf.slice(),
  };
}