/* Analog gauge rendering on <canvas>. */

"use strict";

const Gauge = {
  /* zone builder from a parameter definition.
   * Returns [{lo, hi, color}] in normalised [0,1] domain. */
  zones(param) {
    const p = v => Math.max(0, Math.min(1, (v - param.min) / (param.max - param.min)));
    const z = [];
    if (param.crit_lo !== undefined) z.push({ lo: 0, hi: p(param.crit_lo), color: "#f03a3a" });
    if (param.warn_lo !== undefined) {
      const lo = param.crit_lo !== undefined ? p(param.crit_lo) : 0;
      z.push({ lo, hi: p(param.warn_lo), color: "#f5b83d" });
    }
    const mLo = param.warn_lo !== undefined ? p(param.warn_lo) : 0;
    const mHi = param.warn_hi !== undefined ? p(param.warn_hi) : 1;
    z.push({ lo: mLo, hi: mHi, color: "#29d071" });
    if (param.warn_hi !== undefined) {
      const hi = param.crit_hi !== undefined ? p(param.crit_hi) : 1;
      z.push({ lo: p(param.warn_hi), hi, color: "#f5b83d" });
    }
    if (param.crit_hi !== undefined) z.push({ lo: p(param.crit_hi), hi: 1, color: "#f03a3a" });
    return z;
  },

  draw(canvas, value, param, extra) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H * 0.62;
    const r = Math.min(W, H) * 0.34;
    const a0 = -Math.PI * 0.75, a1 = Math.PI * 0.75; // -135deg .. +135deg
    const frac = Math.max(0, Math.min(1, (value - param.min) / (param.max - param.min)));
    const ang = a0 + frac * (a1 - a0);

    ctx.clearRect(0, 0, W, H);

    // zones
    const zones = Gauge.zones(param);
    const bandW = 13;
    for (const z of zones) {
      const za0 = a0 + z.lo * (a1 - a0), za1 = a0 + z.hi * (a1 - a0);
      ctx.beginPath();
      ctx.arc(cx, cy, r, za0, za1);
      ctx.strokeStyle = z.color; ctx.lineWidth = bandW; ctx.lineCap = "round";
      ctx.globalAlpha = 0.85;
      ctx.stroke(); ctx.globalAlpha = 1;
    }

    // ticks + labels
    ctx.strokeStyle = "#7fa08d"; ctx.fillStyle = "#7fa08d";
    ctx.font = "11px Consolas, monospace"; ctx.textAlign = "center";
    for (let k = 0; k <= 4; k++) {
      const a = a0 + (k / 4) * (a1 - a0);
      const val = param.min + (k / 4) * (param.max - param.min);
      const x1 = cx + Math.cos(a) * (r + 2), y1 = cy + Math.sin(a) * (r + 2);
      const x2 = cx + Math.cos(a) * (r + 9), y2 = cy + Math.sin(a) * (r + 9);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.fillText(fmtVal(val, param), cx + Math.cos(a) * (r + 24), cy + Math.sin(a) * (r + 24) + 4);
    }

    // needle
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(ang) * (r - 6), cy + Math.sin(ang) * (r - 6));
    ctx.strokeStyle = "#e8f5ee"; ctx.lineWidth = 3; ctx.lineCap = "round";
    ctx.stroke();

    // pivot
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
    ctx.fillStyle = "#e8f5ee"; ctx.fill();
    ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
    ctx.fillStyle = "#0b0f0d"; ctx.fill();

    // value readout
    ctx.fillStyle = valueColor(value, param);
    ctx.font = "bold 24px Consolas, monospace";
    ctx.fillText(fmtVal(value, param), cx, cy - r - 20);
    ctx.fillStyle = "#7fa08d";
    ctx.font = "11px Consolas, monospace";
    ctx.fillText((param.unit || ""), cx, cy - r - 6);

    if (extra && extra.label) {
      ctx.fillStyle = "#cfe6d8"; ctx.font = "12px Segoe UI, sans-serif";
      ctx.fillText(extra.label, cx, cy + r + 26);
    }
  }
};

function fmtVal(v, param) {
  const d = (param.decimals !== undefined) ? param.decimals
    : (Math.abs(v) < 10 ? 1 : 0);
  return v.toFixed(d);
}

function valueColor(value, param) {
  if (param.crit_lo !== undefined && value < param.crit_lo) return "#f03a3a";
  if (param.crit_hi !== undefined && value > param.crit_hi) return "#f03a3a";
  if (param.warn_lo !== undefined && value < param.warn_lo) return "#f5b83d";
  if (param.warn_hi !== undefined && value > param.warn_hi) return "#f5b83d";
  return "#29d071";
}