// turntune UI — vanilla JS, no build step.
//
// Flow: GET /api/meta -> render detector selector + sliders + init chart -> on any
// change (debounced) POST /api/sweep + /api/score for the selected detector -> redraw
// the curve (no-fire-aware), the cutoff/latency/no-fire readouts, the bounded headline,
// and the caught-cutoff list.

"use strict";

let META = null;
let DETECTOR = null;
let PARAMS = {};
let SWEEP_AXIS = null;
let CHART = null;
let CURRENT = null; // {x: cutoff%, y: latency, over: bool} operating-point marker
const MAX_ROWS = 50;

const $ = (sel) => document.querySelector(sel);
const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const pctNum = (x, d = 0) => (x == null ? "–" : (x * 100).toFixed(d));

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

const curAxes = () => META.detectors[DETECTOR].axes;
const boundPct = () => (META.no_fire_bound * 100).toFixed(0);

// ---- detector selector + sweep-axis selector + sliders --------------------
function renderControls() {
  $("#meta-sub").textContent =
    `· ${META.dataset} (${META.language}) · ${META.n_scenarios} scenarios`;

  const dsel = $("#detector-select");
  dsel.innerHTML = "";
  for (const name of Object.keys(META.detectors)) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    if (name === DETECTOR) o.selected = true;
    dsel.appendChild(o);
  }
  dsel.onchange = () => {
    DETECTOR = dsel.value;
    PARAMS = { ...META.detectors[DETECTOR].defaults };
    const names = curAxes().map((a) => a.name);
    if (!names.includes(SWEEP_AXIS)) SWEEP_AXIS = META.sweep_axis;
    renderControls();
    refresh();
  };

  const sel = $("#sweep-axis");
  sel.innerHTML = "";
  for (const ax of curAxes()) {
    const o = document.createElement("option");
    o.value = ax.name;
    o.textContent = ax.label;
    if (ax.name === SWEEP_AXIS) o.selected = true;
    sel.appendChild(o);
  }
  sel.onchange = () => { SWEEP_AXIS = sel.value; refresh(); };

  const box = $("#sliders");
  box.innerHTML = "";
  for (const ax of curAxes()) {
    const field = document.createElement("div");
    field.className = "field";
    const decimals = (String(ax.step).split(".")[1] || "").length;
    field.innerHTML = `
      <label>${ax.label}<span class="val" id="val-${ax.name}"></span></label>
      <input type="range" id="rng-${ax.name}" min="${ax.lo}" max="${ax.hi}"
             step="${ax.step}" value="${PARAMS[ax.name]}" />
      <div class="help">${ax.help || ""}</div>`;
    box.appendChild(field);
    const rng = field.querySelector("input");
    const val = field.querySelector(".val");
    const show = () => { val.textContent = Number(PARAMS[ax.name]).toFixed(decimals); };
    show();
    rng.oninput = () => {
      PARAMS[ax.name] = Number(rng.value);
      show();
      refresh();
    };
  }
}

// ---- chart (no-fire aware) ------------------------------------------------
function drawCurrentMarker(u) {
  if (!CURRENT || CURRENT.x == null || CURRENT.y == null) return;
  const cx = u.valToPos(CURRENT.x, "x", true);
  const cy = u.valToPos(CURRENT.y, "y", true);
  const ctx = u.ctx;
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
  ctx.fillStyle = CURRENT.over ? "#dc2626" : "#16a34a"; // red if over no-fire bound
  ctx.fill();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "#fff";
  ctx.stroke();
  ctx.restore();
}

function makeChart() {
  const el = $("#chart");
  const opts = {
    width: el.clientWidth || 560,
    height: 320,
    scales: { x: { time: false }, y: {} },
    axes: [{ label: "cutoff rate (%)" }, { label: "endpointing latency (s)" }],
    series: [
      {},
      { label: "within no-fire", stroke: "#2563eb", width: 0, points: { show: true, size: 7, fill: "#2563eb" } },
      { label: "exceeds no-fire", stroke: "#cbd5e1", width: 0, points: { show: true, size: 7, fill: "#cbd5e1" } },
      { label: "frontier", stroke: "#2563eb", width: 2.5, spanGaps: true, points: { show: false } },
    ],
    cursor: { y: false },
    hooks: { draw: [drawCurrentMarker] },
  };
  CHART = new uPlot(opts, [[], [], [], []], el);
  window.addEventListener("resize", () => CHART.setSize({ width: el.clientWidth || 560, height: 320 }));
}

function drawCurve(sweep) {
  const bound = sweep.no_fire_bound;
  const valid = sweep.points
    .filter((p) => p.p50 != null && p.cutoff != null)
    .map((p) => ({ x: p.cutoff * 100, y: p.p50, nf: p.no_fire ?? 0 }))
    .sort((a, b) => a.x - b.x);
  const xs = valid.map((p) => p.x);
  const yOk = valid.map((p) => (p.nf <= bound ? p.y : null));
  const yOver = valid.map((p) => (p.nf > bound ? p.y : null));
  const parKeys = new Set(
    sweep.pareto.map((p) => `${(p.cutoff * 100).toFixed(2)}|${p.p50.toFixed(3)}`)
  );
  const yPar = valid.map((p) => (parKeys.has(`${p.x.toFixed(2)}|${p.y.toFixed(3)}`) ? p.y : null));

  const c = sweep.current;
  CURRENT = c.cutoff == null || c.p50 == null
    ? null
    : { x: c.cutoff * 100, y: c.p50, over: (c.no_fire ?? 0) > bound };
  CHART.setData([xs, yOk, yOver, yPar]);
  $("#legend-note").innerHTML =
    `<span class="dot ok"></span> within ${boundPct()}% no-fire&nbsp;&nbsp;` +
    `<span class="dot over"></span> exceeds it (bought by silence) — frontier line is drawn only within the bound`;
}

// ---- operating-point readouts + bounded headline --------------------------
function renderOppoint(sweep) {
  const c = sweep.current;
  const over = (c.no_fire ?? 0) > sweep.no_fire_bound;
  $("#oppoint").innerHTML =
    `<div class="stat"><div class="stat-num">${pctNum(c.cutoff)}%</div><div class="stat-lbl">cutoff</div></div>` +
    `<div class="stat"><div class="stat-num">${c.p50 == null ? "n/a" : fmt(c.p50) + "s"}</div><div class="stat-lbl">latency</div></div>` +
    `<div class="stat ${over ? "bad" : ""}"><div class="stat-num">${pctNum(c.no_fire)}%</div><div class="stat-lbl">no-fire</div></div>`;
}

function updateSummary(sweep) {
  const y = boundPct();
  const parts = [
    `<div class="small muted" style="margin-bottom:.25rem">best latency @ ≤X% cutoff <b>and ≤${y}% no-fire</b>:</div>`,
  ];
  for (const b of META.cutoff_budgets) {
    const v = sweep.summary[String(Math.round(b * 100))];
    parts.push(
      `<span class="op">≤${Math.round(b * 100)}% cutoff: <b>${v == null ? "unreachable" : fmt(v) + "s"}</b></span>`
    );
  }
  $("#summary").innerHTML = parts.join(" ");
}

// ---- scoring: counts + caught-cutoff list ---------------------------------
function renderCounts(score) {
  const c = score.counts;
  $("#counts").innerHTML =
    `<span class="chip">✅ correct <b>${c.correct}</b></span>` +
    `<span class="chip">🔴 cutoff <b>${c.cutoff}</b></span>` +
    `<span class="chip">⏳ no-fire <b>${c.missed}</b></span>` +
    `<span class="chip muted">of ${score.n}</span>`;
}

function pct(v, dur) {
  return `${Math.max(0, Math.min(100, (v / dur) * 100))}%`;
}

function renderCutoffs(score) {
  $("#cutoff-count").textContent = `(${score.cutoffs.length})`;
  const list = $("#cutoff-list");
  list.innerHTML = "";
  for (const c of score.cutoffs.slice(0, MAX_ROWS)) {
    const dur = c.duration || 1;
    const li = document.createElement("li");
    const spansHtml = c.spans
      .map(
        (s) =>
          `<div class="span ${s.label}" style="left:${pct(s.start, dur)};width:${pct(s.end - s.start, dur)}"></div>`
      )
      .join("");
    const firedMark = `<div class="mark fired" style="left:${pct(c.fired_s, dur)}" title="fired ${fmt(c.fired_s)}s"></div>`;
    const trueMark = `<div class="mark true" style="left:${pct(c.true_eot_s, dur)}" title="true end ${fmt(c.true_eot_s)}s"></div>`;
    li.innerHTML = `
      <div class="cut-head">
        <span class="id">${c.id}</span> — fired at <b>${fmt(c.fired_s)}s</b>,
        true end of turn at <b>${fmt(c.true_eot_s)}s</b>
        <span class="muted">(${fmt(c.true_eot_s - c.fired_s)}s early)</span>
      </div>
      <div class="timeline">${spansHtml}${trueMark}${firedMark}</div>
      <audio controls preload="none" src="${c.audio}"></audio>`;
    list.appendChild(li);
  }
  if (score.cutoffs.length > MAX_ROWS) {
    const note = document.createElement("li");
    note.className = "muted small";
    note.textContent = `… and ${score.cutoffs.length - MAX_ROWS} more (showing first ${MAX_ROWS}).`;
    list.appendChild(note);
  }
}

// ---- orchestration --------------------------------------------------------
const refresh = debounce(async () => {
  const [sweep, score] = await Promise.all([
    postJSON("/api/sweep", { detector: DETECTOR, params: PARAMS, sweep_axis: SWEEP_AXIS }),
    postJSON("/api/score", { detector: DETECTOR, params: PARAMS }),
  ]);
  drawCurve(sweep);
  renderOppoint(sweep);
  updateSummary(sweep);
  renderCounts(score);
  renderCutoffs(score);
}, 120);

async function main() {
  META = await (await fetch("/api/meta")).json();
  DETECTOR = META.default_detector;
  PARAMS = { ...META.detectors[DETECTOR].defaults };
  SWEEP_AXIS = META.sweep_axis;
  renderControls();
  makeChart();
  refresh();
}

main();
