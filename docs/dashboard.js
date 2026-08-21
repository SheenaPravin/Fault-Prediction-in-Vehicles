/* Dashboard: live telemetry feed + in-browser LSTM inference.
 * - Module cards are clickable and open a detail view (live trends, RUL, events)
 * - Failure-mode detections are logged with mission timestamps
 * - Per-module RUL from the LSTM regression head, fault mode from classifier */

"use strict";

const SPARK_N = 220;          // samples kept per parameter trend
const EVENT_CAP = 400;
const ENTER_N = 8;            // frames a breach must persist before logging
const EXIT_N = 25;            // frames back in band before re-arming a param

const DASH = {
  config: null,
  parts: {},        // pid -> {gaugeParam, hist:{key:[]}, prevStatus:{}, failLatched}
  windowBuf: null,
  events: [],
  clsEpisode: { name: null, armed: true },
  modalPid: null,
};

function $(id) { return document.getElementById(id); }

/* ---------------- parameter value helpers ---------------- */
function scaledValue(param, raw) {
  return (param.scale !== undefined) ? raw * param.scale : raw;
}
function paramStatus(param, raw) {
  const v = scaledValue(param, raw);
  if (param.crit_lo !== undefined && v < param.crit_lo) return "crit";
  if (param.crit_hi !== undefined && v > param.crit_hi) return "crit";
  if (param.warn_lo !== undefined && v < param.warn_lo) return "warn";
  if (param.warn_hi !== undefined && v > param.warn_hi) return "warn";
  return "ok";
}

/* ---------------- build the parts grid ---------------- */
function buildGrid() {
  const grid = $("parts-grid");
  for (const pid of DASH.config.part_order) {
    if (pid === "overall") continue;
    const part = DASH.config.parts[pid];
    const card = document.createElement("section");
    card.className = "panel part-card clickable";
    card.id = "card_" + pid;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", "Open " + part.label + " details");

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
      <span class="open-hint" title="Open details">&#9906;</span>
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

    card.addEventListener("click", () => openModule(pid));
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModule(pid); }
    });

    DASH.parts[pid] = { gaugeParam, hist: {}, prevStatus: {}, statusCount: {},
                        lastStatus: {}, failLatched: false };
    for (const p of part.params) DASH.parts[pid].hist[p.key] = [];
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

/* ---------------- event log ---------------- */
function missionTimeString(rec, cycle) {
  const totalS = rec.time + cycle * LiveFeed.durationS;
  const h = Math.floor(totalS / 3600), m = Math.floor(totalS / 60) % 60,
        s = Math.floor(totalS % 60);
  const pad = x => String(x).padStart(2, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

function addEvent(pid, sev, msg, tstr) {
  const ev = { t: tstr, pid, sev, msg };
  DASH.events.unshift(ev);
  if (DASH.events.length > EVENT_CAP) DASH.events.pop();
  prependEventRow($("eventLog"), ev, true);
  const n = $("eventCount");
  n.textContent = DASH.events.length + " events";
  if (DASH.modalPid) renderModalEvents();
}

function prependEventRow(container, ev, animate) {
  const empty = container.querySelector(".ev-empty");
  if (empty) empty.remove();
  const row = document.createElement("div");
  row.className = "ev-row " + ev.sev + (animate ? " ev-new" : "");
  const mod = ev.pid ? DASH.config.parts[ev.pid].label : "AI MODEL";
  row.innerHTML = `
    <span class="ev-t">${ev.t}</span>
    <span class="ev-mod">${mod}</span>
    <span class="ev-msg">${ev.msg}</span>`;
  container.prepend(row);
  while (container.children.length > EVENT_CAP) container.lastChild.remove();
}

const RANK = { ok: 0, warn: 1, crit: 2 };

function detectParamEvents(pid, rec, health, tstr) {
  const st = DASH.parts[pid];
  const part = DASH.config.parts[pid];
  for (const p of part.params) {
    const raw = rec[p.key];
    if (raw === undefined) continue;
    const s = paramStatus(p, raw);
    const prev = st.prevStatus[p.key] || "ok";
    if (s === st.lastStatus[p.key]) st.statusCount[p.key] += 1;
    else st.statusCount[p.key] = 1;
    st.lastStatus[p.key] = s;
    const cnt = st.statusCount[p.key];

    if (cnt === ENTER_N && RANK[s] > RANK[prev]) {
      if (s === "crit") {
        addEvent(pid, "crit",
          `${p.label} CRITICAL — ${fmtVal(scaledValue(p, raw), p)} ${p.unit}`, tstr);
      } else {
        addEvent(pid, "warn",
          `${p.label} entered warning band — ${fmtVal(scaledValue(p, raw), p)} ${p.unit}`, tstr);
      }
      st.prevStatus[p.key] = s;
    } else if (RANK[s] < RANK[prev] && cnt >= EXIT_N) {
      st.prevStatus[p.key] = s;   // recovered long enough; re-arm
    }
  }
  if (!st.failLatched && health < DASH.config.fail_health) {
    st.failLatched = true;
    addEvent(pid, "fail", `health ${health.toFixed(0)} — failure threshold crossed`, tstr);
  } else if (st.failLatched && health > DASH.config.fail_health + 10) {
    st.failLatched = false;
  }
}

function detectAiEvent(cls, tstr) {
  const names = DASH.config.class_names;
  let top = 0;
  for (let i = 1; i < cls.length; i++) if (cls[i] > cls[top]) top = i;
  const name = names[top], prob = cls[top];
  const ep = DASH.clsEpisode;
  if (top !== 0 && ep.armed && prob >= 0.5) {
    addEvent(null, "ai",
      `fault mode "${name.replace(/_/g, " ")}" detected (${Math.round(prob * 100)}% confidence)`, tstr);
    ep.armed = false;
    ep.name = name;
  } else if (top === 0 || prob < 0.3) {
    ep.armed = true;
    if (top === 0) ep.name = null;
  }
}

/* ---------------- per-part update ---------------- */
function updatePart(pid, rec, healthVal, rulFrac) {
  const part = DASH.config.parts[pid];
  const st = DASH.parts[pid];
  const card = $("card_" + pid);

  const hEl = $("health_" + pid);
  hEl.textContent = healthVal.toFixed(0);
  hEl.className = healthVal < DASH.config.fail_health ? "crit" : (healthVal < 50 ? "warn" : "");

  const rulPct = Math.max(0, Math.min(100, rulFrac * 100));
  const rulBar = $("rulbar_" + pid);
  rulBar.style.width = rulPct + "%";
  rulBar.className = rulPct < 15 ? "red" : (rulPct < 40 ? "amber" : "green");
  $("rultext_" + pid).textContent = rulText(rulFrac);

  let worst = "ok";
  for (const p of part.params) {
    const raw = rec[p.key];
    if (raw === undefined) continue;
    const val = scaledValue(p, raw);
    const stat = paramStatus(p, raw);
    if (stat === "crit") worst = "crit"; else if (stat === "warn" && worst !== "crit") worst = "warn";

    const vEl = $("val_" + pid + "_" + p.key);
    vEl.textContent = fmtVal(val, p) + (p.unit ? " " + p.unit : "");
    vEl.className = "p-val " + stat;
    renderTrack(pid, p, raw);

    const hist = st.hist[p.key];
    hist.push(val);
    if (hist.length > SPARK_N) hist.shift();
  }

  const gp = st.gaugeParam;
  const gval = rec[gp.key] !== undefined ? scaledValue(gp, rec[gp.key]) : gp.min;
  Gauge.draw($("gauge_" + pid), gval, gp, { label: gp.label });

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

/* ---------------- overall + status ---------------- */
function updateOverall(rec, reg, cls, overallHealth) {
  const overallFrac = reg[DASH.config.part_order.indexOf("overall")];

  Gauge.draw($("gauge_overall"), overallHealth, DASH.config.parts.overall.params[0],
             { label: "Fused Health Index" });
  $("rulOverall").textContent = rulText(overallFrac);
  $("rulOverallPct").className = "m-sub " +
    (overallFrac < 0.15 ? "crit" : (overallFrac < 0.4 ? "warn" : ""));

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

  $("v_rpm").textContent = rec.rpm.toFixed(0);
  $("v_load").textContent = (rec.load * 100).toFixed(0) + "%";
  $("v_terrain").textContent = (rec.terrain * 100).toFixed(0) + "%";
  const oilp = rec.oil_pressure * 1e-5, oilt = rec.oil_temp;
  const opEl = $("v_oilp");
  opEl.textContent = oilp.toFixed(1) + " bar";
  opEl.className = "digital small " + (oilp < 2 ? "crit" : (oilp < 3.2 ? "warn" : ""));
  const otEl = $("v_oilt");
  otEl.textContent = oilt.toFixed(0) + "°C";
  otEl.className = "digital small " + (oilt > 130 ? "crit" : (oilt > 110 ? "warn" : ""));
  const vrEl = $("v_vibrms");
  vrEl.textContent = rec.vib_rms.toFixed(2);
  vrEl.className = "digital small " + (rec.vib_rms > 1.2 ? "crit" : (rec.vib_rms > 0.75 ? "warn" : ""));

  const worstPart = Math.min(...Object.keys(DASH.parts)
    .map(pid => LiveFeed.stream.health[pid][LiveFeed.idx]));
  let status;
  if (overallHealth < DASH.config.fail_health || overallFrac < 0.03 || worstPart < DASH.config.fail_health) {
    status = "sos";
  } else if (overallHealth < 40 || overallFrac < 0.2) {
    status = "faulty";
  } else if (overallHealth < 60 || overallFrac < 0.45 || worstPart < 50) {
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
      const h = LiveFeed.stream.health[pid][LiveFeed.idx];
      if (h < DASH.config.fail_health) detail.push(DASH.config.parts[pid].label);
    }
    banner.classList.remove("hidden");
    $("sosDetail").textContent = detail.length
      ? " Modules: " + detail.join(", ") + " — immediate maintenance required."
      : " Overall health below failure threshold — immediate maintenance required.";
  } else {
    banner.classList.add("hidden");
  }
}

/* ---------------- module detail modal ---------------- */
function openModule(pid) {
  DASH.modalPid = pid;
  const part = DASH.config.parts[pid];
  $("modalTitle").textContent = part.label;
  $("modalSub").textContent = part.params.length + " monitored parameters · live feed";

  const gaugeParam = DASH.parts[pid].gaugeParam;
  Gauge.draw($("gauge_modal"), gaugeParam.min, gaugeParam, { label: gaugeParam.label });

  const list = $("modalParams");
  list.innerHTML = "";
  for (const p of part.params) {
    const item = document.createElement("div");
    item.className = "spark-item";
    item.innerHTML = `
      <div class="spark-head">
        <span class="p-label">${p.label}</span>
        <span class="p-val" id="mspark_val_${p.key}">--</span>
        <span class="p-unit">${p.unit}</span>
      </div>
      <canvas class="spark" id="mspark_${p.key}" width="460" height="54"></canvas>`;
    list.appendChild(item);
  }
  renderModalEvents();
  $("moduleModal").classList.remove("hidden");
  updateModal();
}

function closeModule() {
  DASH.modalPid = null;
  $("moduleModal").classList.add("hidden");
}

function renderModalEvents() {
  const box = $("modalEvents");
  box.innerHTML = "";
  const evs = DASH.events.filter(e => e.pid === DASH.modalPid).slice(0, 30);
  if (!evs.length) {
    box.innerHTML = '<div class="ev-empty">No failure modes detected yet.</div>';
    return;
  }
  for (const ev of evs) prependEventRow(box, ev, false);
}

function drawSpark(canvas, data, p) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0a120e";
  ctx.fillRect(0, 0, W, H);
  if (!data || data.length < 2) return;

  const norm = v => Math.max(0, Math.min(1, (v - p.min) / (p.max - p.min)));
  const yOf = v => H - 4 - norm(v) * (H - 8);

  // threshold guides
  ctx.setLineDash([4, 4]);
  for (const [th, color] of [[p.warn_hi, "#f5b83d"], [p.warn_lo, "#f5b83d"],
                             [p.crit_hi, "#f03a3a"], [p.crit_lo, "#f03a3a"]]) {
    if (th === undefined || th < p.min || th > p.max) continue;
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    ctx.moveTo(0, yOf(th));
    ctx.lineTo(W, yOf(th));
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  // trace
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = i / (SPARK_N - 1) * W;
    const y = yOf(data[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#3dd6f0";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  // endpoint
  const last = data[data.length - 1];
  ctx.beginPath();
  ctx.arc(W - 2, yOf(last), 3, 0, 2 * Math.PI);
  ctx.fillStyle = valueColor(last, p);
  ctx.fill();
}

function updateModal() {
  const pid = DASH.modalPid;
  if (!pid) return;
  const rec = LiveFeed.records[LiveFeed.idx];
  const idx = LiveFeed.idx;
  const health = LiveFeed.stream.health[pid][idx];
  const j = DASH.config.part_order.indexOf(pid);
  const rulFrac = DASH.lastReg ? DASH.lastReg[j] : 1;

  const hEl = $("modalHealth");
  hEl.textContent = health.toFixed(0);
  hEl.className = health < DASH.config.fail_health ? "crit" : (health < 50 ? "warn" : "");

  const rulPct = Math.max(0, Math.min(100, rulFrac * 100));
  const bar = $("modalRulBar");
  bar.style.width = rulPct + "%";
  bar.className = rulPct < 15 ? "red" : (rulPct < 40 ? "amber" : "green");
  $("modalRulText").textContent = rulText(rulFrac);
  const badge = $("modalRulBadge");
  badge.className = "badge " + (rulPct < 15 ? "sos" : (rulPct < 40 ? "degraded" : "healthy"));
  badge.textContent = "RUL " + (rulFrac * 100).toFixed(0) + "%";

  const st = DASH.parts[pid];
  const gp = st.gaugeParam;
  const gval = rec[gp.key] !== undefined ? scaledValue(gp, rec[gp.key]) : gp.min;
  Gauge.draw($("gauge_modal"), gval, gp, { label: gp.label });

  for (const p of DASH.config.parts[pid].params) {
    const raw = rec[p.key];
    if (raw === undefined) continue;
    const val = scaledValue(p, raw);
    const vEl = $("mspark_val_" + p.key);
    if (vEl) {
      vEl.textContent = fmtVal(val, p) + " " + p.unit;
      vEl.className = "p-val " + paramStatus(p, raw);
    }
    drawSpark($("mspark_" + p.key), st.hist[p.key], p);
  }
}

/* ---------------- live frame ---------------- */
function onFrame(rec, idx, cycle) {
  const vec = normaliseFeatures(rec);
  const win = DASH.windowBuf.push(vec);
  const { reg, cls } = LSTM.forward(DASH.model, win);
  DASH.lastReg = reg;
  DASH.lastCls = cls;

  const tstr = missionTimeString(rec, cycle);
  let worstOverall = "ok";
  for (const pid of DASH.config.part_order) {
    if (pid === "overall") continue;
    const j = DASH.config.part_order.indexOf(pid);
    const health = LiveFeed.stream.health[pid][idx];
    const w = updatePart(pid, rec, health, reg[j]);
    detectParamEvents(pid, rec, health, tstr);
    if (w === "crit") worstOverall = "crit"; else if (w === "warn" && worstOverall !== "crit") worstOverall = "warn";
  }
  const overallHealth = LiveFeed.stream.health.overall[idx];
  updateOverall(rec, reg, cls, overallHealth);
  detectAiEvent(cls, tstr);
  updateModal();

  $("missionClock").textContent = tstr;
  $("cycleInfo").textContent = `pass ${cycle + 1} · ${DASH.config.dt.toFixed(2)}s sampling · ${LiveFeed.speed}× speed`;
}

function onCycleWrap(cycle) {
  // restart LSTM window and detection latches for the new pass
  const first = normaliseFeatures(LiveFeed.records[0]);
  DASH.windowBuf = rollingWindow(DASH.config.window, DASH.config.input_features.length, first);
  DASH.clsEpisode = { name: null, armed: true };
  for (const pid of Object.keys(DASH.parts)) {
    DASH.parts[pid].prevStatus = {};
    DASH.parts[pid].statusCount = {};
    DASH.parts[pid].lastStatus = {};
    DASH.parts[pid].failLatched = false;
  }
}

/* ---------------- init ---------------- */
async function init() {
  const [config, model] = await Promise.all([
    fetch("config.json").then(r => r.json()),
    fetch("model.json").then(r => r.json()),
  ]);
  DASH.config = config;
  DASH.model = model;
  await LiveFeed.init(config);

  DASH.windowBuf = rollingWindow(config.window, config.input_features.length,
                                 normaliseFeatures(LiveFeed.records[0]));

  buildGrid();
  $("missionName").textContent = "Mission: " + LiveFeed.stream.meta.name.replace(/_/g, " ");
  $("injectedFaults").innerHTML = (LiveFeed.stream.meta.faults || []).map(f =>
    `<span class="chip">&#9889; ${f.replace(/_/g, " ")}</span>`).join("");

  $("modelInfo").textContent =
    `LSTM · ${model.H} hidden units · input window ${config.window}×${config.input_features.length} · ` +
    `RUL cap ${config.rul_cap_steps} steps · trained offline on physics-simulated scenarios, inference runs live in-browser`;

  $("btnPlay").onclick = () => {
    if (LiveFeed.playing) { LiveFeed.pause(); $("btnPlay").textContent = "▶"; }
    else { LiveFeed.play(); $("btnPlay").textContent = "❚❚"; }
  };
  document.querySelectorAll(".speed-btn").forEach(b => {
    b.onclick = () => {
      LiveFeed.setSpeed(Number(b.dataset.speed));
      document.querySelectorAll(".speed-btn").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
    };
  });
  $("btnCloseModal").onclick = closeModule;
  $("modalBackdrop").onclick = closeModule;
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeModule(); });

  LiveFeed.onRecord(onFrame);
  LiveFeed.onCycle(onCycleWrap);
  LiveFeed.play();
}

init();
