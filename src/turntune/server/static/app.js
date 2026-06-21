// turntune UI — vanilla JS, no build step.
//
// Flow: GET /api/meta -> render sliders + init chart -> on any change (debounced)
// POST /api/sweep + /api/score -> redraw the latency-vs-cutoff curve, the operating
// point, the headline numbers, and the caught-cutoff list with audio + timeline.

"use strict";

let META = null;
let PARAMS = {};
let SWEEP_AXIS = null;
let CHART = null;
let CURRENT = null; // {x: cutoff%, y: latency} operating-point marker
const MAX_ROWS = 50;

const $ = (sel) => document.querySelector(sel);
const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));

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

// ---- sliders + sweep-axis selector ----------------------------------------
function renderControls() {
  $("#meta-sub").textContent =
    `· ${META.detector} · ${META.dataset} (${META.language}) · ${META.n_scenarios} scenarios`;

  const sel = $("#sweep-axis");
  sel.innerHTML = "";
  for (const ax of META.axes) {
    const o = document.createElement("option");
    o.value = ax.name;
    o.textContent = ax.label;
    if (ax.name === SWEEP_AXIS) o.selected = true;
    sel.appendChild(o);
  }
  sel.onchange = () => { SWEEP_AXIS = sel.value; refresh(); };

  const box = $("#sliders");
  box.innerHTML = "";
  for (const ax of META.axes) {
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

// ---- chart ----------------------------------------------------------------
function drawCurrentMarker(u) {
  if (!CURRENT || CURRENT.x == null || CURRENT.y == null) return;
  const cx = u.valToPos(CURRENT.x, "x", true);
  const cy = u.valToPos(CURRENT.y, "y", true);
  const ctx = u.ctx;
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
  ctx.fillStyle = "#dc2626";
  ctx.fill();
  ctx.lineWidth = 2;
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
      { label: "sweep", stroke: "#9ca3af", width: 1, points: { show: true, size: 5 } },
      { label: "pareto", stroke: "#2563eb", width: 2.5, spanGaps: true, points: { show: true, size: 5 } },
    ],
    cursor: { y: false },
    hooks: { draw: [drawCurrentMarker] },
  };
  CHART = new uPlot(opts, [[], [], []], el);
  window.addEventListener("resize", () => CHART.setSize({ width: el.clientWidth || 560, height: 320 }));
}

function drawCurve(sweep) {
  const valid = sweep.points
    .filter((p) => p.p50 != null && p.cutoff != null)
    .map((p) => ({ x: p.cutoff * 100, y: p.p50 }))
    .sort((a, b) => a.x - b.x);
  const xs = valid.map((p) => p.x);
  const ySweep = valid.map((p) => p.y);
  const parKeys = new Set(
    sweep.pareto.map((p) => `${(p.cutoff * 100).toFixed(2)}|${p.p50.toFixed(3)}`)
  );
  const yPar = valid.map((p) => (parKeys.has(`${p.x.toFixed(2)}|${p.y.toFixed(3)}`) ? p.y : null));
  CURRENT = sweep.current.cutoff == null || sweep.current.p50 == null
    ? null
    : { x: sweep.current.cutoff * 100, y: sweep.current.p50 };
  CHART.setData([xs, ySweep, yPar]);
}

function updateSummary(sweep) {
  const parts = [];
  for (const b of META.cutoff_budgets) {
    const key = String(Math.round(b * 100));
    const v = sweep.summary[key];
    parts.push(
      `<span class="op">latency @ ≤${key}% cutoff: <b>${v == null ? "unreachable" : fmt(v) + "s"}</b></span>`
    );
  }
  const c = sweep.current;
  const cur = c.cutoff == null
    ? ""
    : `<div class="small muted" style="margin-top:.4rem">current operating point: ` +
      `cutoff <b>${fmt(c.cutoff * 100, 0)}%</b>, latency <b>${c.p50 == null ? "n/a" : fmt(c.p50) + "s"}</b>` +
      `${c.no_fire ? `, no-fire ${fmt(c.no_fire * 100, 0)}%` : ""}</div>`;
  $("#summary").innerHTML = parts.join(" ") + cur;
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
    postJSON("/api/sweep", { params: PARAMS, sweep_axis: SWEEP_AXIS }),
    postJSON("/api/score", { params: PARAMS }),
  ]);
  drawCurve(sweep);
  updateSummary(sweep);
  renderCounts(score);
  renderCutoffs(score);
}, 120);

async function main() {
  META = await (await fetch("/api/meta")).json();
  PARAMS = { ...META.defaults };
  SWEEP_AXIS = META.sweep_axis;
  renderControls();
  makeChart();
  refresh();
}

main();
