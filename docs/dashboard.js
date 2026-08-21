/* Dashboard: loads the trained LSTM + physics config + demo telemetry,
 * replays the live stream, runs the LSTM in the browser for per-part and
 * overall RUL regression, classifies the fault mode and raises SOS. */

"use strict";

const DASH = {
  config: null, model: null, stream: null,
  parts: {},          // partId -> rendered card data
  idx: 0, playing: true, speed: 1,
  windowBuf: null,    // rolling LSTM input window
  sos: false,
};

function $(id) { return document.getElementById(id); }

/* ---------------- parameter value helpers ---------------- */
function scaledValue(param, raw) {
  const v = (param.scale !== undefined) ? raw * param.scale : raw;
  return v;
}
function paramStatus(param, raw) {
  const v = scaledValue(param, raw);
  if (param.crit_lo !== undefined && v < param.crit_lo) return "crit";
  if (param.crit_hi !== undefined && v > param.crit_hi) return "crit";
  if (param.warn_lo !== undefined && v < param.warn_lo) return "warn";
  if (param.warn_hi !== undefined && v > param.warn_hi) return "warn";
  return "ok";
}
function fmtVal(v, param) {
  const d = (param.decimals !== undefined) ? param.decimals
    : (Math.abs(v) < 10 ? 1 : 0);
  return v.toFixed(d);
}

/* ---------------- build the parts grid ---------------- */
function buildGrid() {
  const grid = $("parts-grid");
  for (const pid of DASH.config.part_order) {
    if (pid === "overall") continue;
    const part = DASH.config.parts[pid];
    const card = document.createElement("section");
    card.className = "panel part-card";
    card.id = "card_" + pid;

    const gaugeParam = part.params.find(p => p.key === part.gauge);

    let rows = "";
    for (const p of part.params) {
      rows += `
        <div class="param-row">
          <span class="p-label">${p.label}</span>
          <span class="p-val" id="val_${pid}_${p.key}">--</span>
          <span class="p-unit">${p.unit}</span>
        </div>
        <div class="param-track" id="track_${pid}_${p.key}"></div>`;
    }

    card.innerHTML = `
      <div class="part-meta">
        <span class="p-name">${part.label}</span>
        <span class="p-health">HEALTH <b id="health_${pid}">--</b>/100</span>
      </div>
      <div class="part-body">
        <div class="gauge-wrap">
          <canvas id="gauge_${pid}" width="170" height="150"></canvas>
        </div>
        <div class="params">
          <div class="metric part-rul">
            <span class="m-label">LSTM RUL — ${part.label}</span>
            <div class="rul-bar"><i id="rulbar_${pid}" class="green" style="width:0%"></i></div>
            <span class="rul-text" id="rultext_${pid}">--</span>
          </div>
          ${rows}
        </div>
      </div>`;
    grid.appendChild(card);

    DASH.parts[pid] = { gaugeParam };
    Gauge.draw($("gauge_" + pid), gaugeParam.min, gaugeParam, { label: gaugeParam.label });
  }
}

/* ---------------- track (threshold bar) rendering ---------------- */
function renderTrack(pid, p, raw) {
  const track = $("track_" + pid + "_" + p.key);
  if (!track) return;
  track.innerHTML = "";
  const pct = v => Math.max(0, Math.min(100, (v - p.min) / (p.max - p.min) * 100));
  const zones = [
    [p.crit_lo, p.warn_lo, "#f03a3a"],
    [p.warn_lo, p.warn_hi, "#f5b83d"],
    [p.warn_hi, p.crit_hi, "#f5b83d"],
  ];
  for (const [lo, hi, color] of zones) {
    if (lo === undefined || hi === undefined) continue;
    const div = document.createElement("div");
    div.className = "zone";
    div.style.left = pct(Math.min(lo, hi)) + "%";
    div.style.width = Math.max(0, pct(Math.max(lo, hi)) - pct(Math.min(lo, hi))) + "%";
    div.style.background = color;
    track.appendChild(div);
  }
  const cur = document.createElement("div");
  cur.className = "cur";
  cur.style.left = pct(scaledValue(p, raw)) + "%";
  track.appendChild(cur);
}

/* ---------------- per-part update ---------------- */
function updatePart(pid, rec, healthVal, rulFrac) {
  const part = DASH.config.parts[pid];
  const card = $("card_" + pid);

  // health
  const hEl = $("health_" + pid);
  hEl.textContent = healthVal.toFixed(0);
  hEl.className = healthVal < DASH.config.fail_health ? "crit" : (healthVal < 50 ? "warn" : "");

  // RUL bar
  const rulPct = Math.max(0, Math.min(100, rulFrac * 100));
  const rulBar = $("rulbar_" + pid);
  rulBar.style.width = rulPct + "%";
  rulBar.className = rulPct < 15 ? "red" : (rulPct < 40 ? "amber" : "green");
  $("rultext_" + pid).textContent = rulText(rulFrac);

  // params
  let worst = "ok";
  for (const p of part.params) {
    const raw = rec[p.key];
    if (raw === undefined) continue;
    const val = scaledValue(p, raw);
    const st = paramStatus(p, raw);
    if (st === "crit") worst = "crit"; else if (st === "warn" && worst !== "crit") worst = "warn";

    const vEl = $("val_" + pid + "_" + p.key);
    vEl.textContent = fmtVal(val, p) + (p.unit ? " " + p.unit : "");
    vEl.className = "p-val " + st;
    renderTrack(pid, p, raw);
  }

  // gauge
  const gp = DASH.parts[pid].gaugeParam;
  const gval = rec[gp.key] !== undefined ? scaledValue(gp, rec[gp.key]) : gp.min;
  Gauge.draw($("gauge_" + pid), gval, gp, { label: gp.label });

  // card frame state
  card.classList.remove("sos", "warn");
  if (worst === "crit" || healthVal < DASH.config.fail_health || rulFrac < 0.05) {
    card.classList.add("sos");
  } else if (worst === "warn" || healthVal < 50 || rulFrac < 0.3) {
    card.classList.add("warn");
  }
  return worst;
}

function rulText(frac) {
  const steps = frac * DASH.config.rul_cap_steps;
  const hours = steps * DASH.config.dt / 3600;
  const pct = Math.max(0, Math.min(100, frac * 100));
  if (hours >= 2) return `≈ ${hours.toFixed(1)} h remaining  (${pct.toFixed(0)}% life)`;
  if (hours >= 0.1) return `≈ ${(hours * 60).toFixed(0)} min remaining  (${pct.toFixed(0)}% life)`;
  return `CRITICAL — ${(hours * 3600).toFixed(0)} s remaining`;
}

/* ---------------- LSTM prediction ---------------- */
function normaliseFeatures(rec) {
  const feat = [];
  for (const key of DASH.config.input_features) {
    const v = rec[key];
    const s = DASH.config.scaler[key];
    feat.push(Math.max(0, Math.min(1, (v - s.min) / Math.max(s.max - s.min, 1e-9))));
  }
  return feat;
}

let lastRul = null, lastCls = null;
function predict(idx) {
  const rec = DASH.stream.records[idx];
  const vec = normaliseFeatures(rec);
  const win = DASH.windowBuf.push(vec);
  const { reg, cls } = LSTM.forward(DASH.model, win);
  lastRul = reg;
  lastCls = cls;
  return { reg, cls };
}

/* ---------------- overall + status ---------------- */
function updateOverall(rec, reg, cls) {
  const health = rec.health_index;
  const overallFrac = reg[DASH.config.part_order.indexOf("overall")];

  Gauge.draw($("gauge_overall"), health, DASH.config.parts.overall.params[0],
             { label: "Fused Health Index" });
  $("rulOverall").textContent = rulText(overallFrac);
  $("rulOverallPct").className = "digital small " +
    (overallFrac < 0.15 ? "crit" : (overallFrac < 0.4 ? "warn" : ""));

  // fault-class probability bars (top 3)
  const names = DASH.config.class_names;
  const order = names.map((n, i) => ({ n, p: cls[i] }))
    .filter(x => x.p > 0.01)
    .sort((a, b) => b.p - a.p).slice(0, 3);
  const bars = $("clsBars");
  bars.innerHTML = "";
  for (const o of order) {
    const row = document.createElement("div");
    row.className = "cls-row";
    row.innerHTML = `
      <span class="cls-name">${o.n.replace(/_/g, " ")}</span>
      <span class="cls-bar"><i style="width:${(o.p * 100).toFixed(0)}%"></i></span>
      <span class="cls-pct">${(o.p * 100).toFixed(0)}%</span>`;
    bars.appendChild(row);
  }

  // vitals
  $("v_rpm").textContent = rec.rpm.toFixed(0);
  $("v_load").textContent = (rec.load * 100).toFixed(0) + "%";
  $("v_terrain").textContent = (rec.terrain * 100).toFixed(0) + "%";
  const oilp = rec.oil_pressure * 1e-5, oilt = rec.oil_temp;
  const opEl = $("v_oilp");
  opEl.textContent = oilp.toFixed(1) + " bar";
  opEl.className = "digital small " + (oilp < 2 ? "crit" : (oilp < 3.2 ? "warn" : ""));
  const otEl = $("v_oilt");
  otEl.textContent = oilt.toFixed(0) + "°C";
  otEl.className = "digital small " + (oilt > 135 ? "crit" : (oilt > 115 ? "warn" : ""));
  const vrEl = $("v_vibrms");
  vrEl.textContent = rec.vib_rms.toFixed(2);
  vrEl.className = "digital small " + (rec.vib_rms > 1.2 ? "crit" : (rec.vib_rms > 0.75 ? "warn" : ""));

  // status
  const worstPart = Math.min(...Object.keys(DASH.parts)
    .map(pid => DASH.stream.health[pid][DASH.idx]));
  let status;
  if (health < DASH.config.fail_health || overallFrac < 0.03 || worstPart < DASH.config.fail_health) {
    status = "sos";
  } else if (health < 40 || overallFrac < 0.2) {
    status = "faulty";
  } else if (health < 60 || overallFrac < 0.45 || worstPart < 50) {
    status = "degraded";
  } else {
    status = "healthy";
  }
  setStatus(status, rec);
  return status;
}

function setStatus(status, rec) {
  const badge = $("statusBadge");
  const mode = $("overallMode");
  const cls = status === "sos" ? "sos" : status;
  badge.className = "badge " + cls;
  badge.textContent = status.toUpperCase();
  mode.className = "badge " + cls;
  mode.textContent = status.toUpperCase();

  const banner = $("sosBanner");
  const detail = [];
  if (status === "sos") {
    for (const pid of DASH.config.part_order) {
      if (pid === "overall") continue;
      const h = DASH.stream.health[pid][DASH.idx];
      if (h < DASH.config.fail_health) detail.push(DASH.config.parts[pid].label);
    }
    banner.classList.remove("hidden");
    $("sosDetail").textContent = detail.length
      ? "Parts: " + detail.join(", ") + " — immediate maintenance required."
      : "Overall health below failure threshold — immediate maintenance required.";
  } else {
    banner.classList.add("hidden");
  }
}

/* ---------------- stream loop ---------------- */
function tick() {
  if (!DASH.stream || !DASH.playing) return;
  const n = DASH.stream.records.length;
  DASH.idx = Math.min(DASH.idx + DASH.speed, n - 1);
  renderFrame();
}

function renderFrame() {
  const rec = DASH.stream.records[DASH.idx];
  const { reg, cls } = predict(DASH.idx);

  let worstOverall = "ok";
  for (const pid of DASH.config.part_order) {
    if (pid === "overall") continue;
    const j = DASH.config.part_order.indexOf(pid);
    const w = updatePart(pid, rec, DASH.stream.health[pid][DASH.idx], reg[j]);
    if (w === "crit") worstOverall = "crit"; else if (w === "warn" && worstOverall !== "crit") worstOverall = "warn";
  }
  updateOverall(rec, reg, cls);

  // clock + progress
  const totalS = DASH.idx * DASH.config.dt;
  const mm = String(Math.floor(totalS / 60)).padStart(2, "0");
  const ss = String(Math.floor(totalS % 60)).padStart(2, "0");
  $("missionClock").textContent = mm + ":" + ss;
  $("missionBar").style.width = (DASH.idx / (DASH.stream.records.length - 1) * 100) + "%";
  $("scrubber").value = DASH.idx / (DASH.stream.records.length - 1) * 100;
}

/* ---------------- init ---------------- */
async function init() {
  const [config, model, stream] = await Promise.all([
    fetch("config.json").then(r => r.json()),
    fetch("model.json").then(r => r.json()),
    fetch("live_stream.json").then(r => r.json()),
  ]);
  DASH.config = config;
  DASH.model = model;
  DASH.stream = stream;

  DASH.windowBuf = rollingWindow(config.window, config.input_features.length,
                                 normaliseFeatures(stream.records[0]));

  buildGrid();
  $("missionName").textContent = "Mission: " + stream.meta.name.replace(/_/g, " ");
  const chips = $("injectedFaults");
  chips.innerHTML = (stream.meta.faults || []).map(f =>
    `<span class="chip">⚡ ${f.replace(/_/g, " ")}</span>`).join("");

  $("modelInfo").textContent =
    `LSTM · ${model.H} hidden units · input window ${config.window}×${config.input_features.length} · ` +
    `RUL cap ${config.rul_cap_steps} steps · trained offline on physics-simulated scenarios (numpy), inference runs in-browser`;

  // controls
  $("btnPlay").onclick = () => {
    DASH.playing = !DASH.playing;
    $("btnPlay").textContent = DASH.playing ? "❚❚" : "▶";
  };
  document.querySelectorAll(".speed-btn").forEach(b => {
    b.onclick = () => {
      DASH.speed = Number(b.dataset.speed);
      document.querySelectorAll(".speed-btn").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
    };
  });
  $("scrubber").oninput = e => {
    DASH.idx = Math.round(Number(e.target.value) / 100 * (stream.records.length - 1));
    DASH.windowBuf = rollingWindow(config.window, config.input_features.length,
                                   normaliseFeatures(stream.records[0]));
    for (let i = Math.max(0, DASH.idx - config.window + 1); i <= DASH.idx; i++) {
      DASH.windowBuf.push(normaliseFeatures(stream.records[i]));
    }
    renderFrame();
  };

  renderFrame();
  setInterval(tick, 66);
}

init();