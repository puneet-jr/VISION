"use strict";

/* ====================== helpers ====================== */
const $ = (id) => document.getElementById(id);
const PAGE_SIZE = 15;
const MAX_ALERTS = 40;

const BEHAVIOR_LABEL = {
  intrusion: "Inside restricted zone", loitering: "Loitering", running: "Moving fast",
  approaching_zone: "Approaching zone", unattended_object: "Unattended item", crowding: "Crowding",
};
const BEHAVIOR_CLASS = { loitering: "bad", unattended_object: "bad", crowding: "warn", intrusion: "warn", running: "warn" };
const FACTOR_LABEL = { object_risk: "Object risk", confidence: "Detector confidence", movement: "Movement",
  proximity: "Zone proximity", persistence: "Persistence" };
const BEHAVIOR_TERM = { intrusion: "b_intrusion", loitering: "b_loitering", running: "b_running",
  approaching_zone: "b_approaching", unattended_object: "b_unattended", crowding: "b_crowding" };
const FACTOR_TERM = { object_risk: "object_risk", confidence: "f_confidence", movement: "f_movement",
  proximity: "f_proximity", persistence: "f_persistence" };
const SOURCE_HINT = {
  video: "Plays a file from data/videos at analysis speed. Use Upload video to add your own.",
  browser: "Streams a phone or laptop camera from a web page. Phones need HTTPS - start the server with scripts/run_demo.py.",
  camera: "Uses the webcam connected to this computer (CAMERA_INDEX in .env).",
  stream: "Uses the IP-camera stream URL configured in .env (it is never typed here).",
  image: "Analyses one still image. Movement, dwell time and behaviours need video, so they show as unknown.",
};
const STATE_LABEL = { idle: "Idle", running: "Processing", stopped: "Stopped", completed: "Video finished", failed: "Source error" };

const state = {
  sessionId: null, running: false, page: 1, ws: null, retry: 0, view: "compare",
  config: null, zones: { reference_size: [1280, 720], zones: [] },
  objects: [], selected: null, seenIds: new Set(), seenSession: null, alerts: [], alertCount: 0,
  network: null, feedKey: "",
};

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null) continue;
    if (k === "class") node.className = v; else if (k === "text") node.textContent = v; else node.setAttribute(k, v);
  }
  children.forEach((c) => c != null && node.append(c));
  return node;
}
const pill = (level) => level ? el("span", { class: `lvl ${level}`, text: level }) : document.createTextNode("-");
const fmtTime = (iso) => { const d = new Date(iso); return isNaN(d) ? "-" : d.toLocaleTimeString(); };
const fmtDateTime = (iso) => { const d = new Date(iso); return isNaN(d) ? "-" : d.toLocaleString(); };
const fmt1 = (n) => (n == null ? "-" : Number(n).toFixed(1));
function showError(msg) { const b = $("errorBanner"); b.textContent = msg; b.hidden = !msg; }
function setBadge(node, text, cls) { node.textContent = text; node.className = `badge ${cls}`; }
function chip(code) {
  const c = el("span", { class: `chip ${BEHAVIOR_CLASS[code] || ""}`, text: BEHAVIOR_LABEL[code] || code });
  const g = GLOSSARY[BEHAVIOR_TERM[code]]; if (g) c.title = g.text;
  return c;
}
function helpIcon(term) {
  const g = GLOSSARY[term]; if (!g) return document.createTextNode("");
  return el("span", { class: "help", "data-tip": `${g.title}: ${g.text}`, tabindex: "0", role: "note", text: "?" });
}

async function api(path, opts = {}) {
  let res;
  try { res = await fetch(path, opts); } catch (_) { throw new Error("Cannot reach the server."); }
  let body = null;
  try { body = await res.json(); } catch (_) { /* no JSON body */ }
  if (!res.ok) throw new Error(body?.error?.message || body?.detail || `Request failed (${res.status})`);
  return body;
}
const jsonOpts = (method, data) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

/* ====================== video feeds ====================== */
const imgBefore = $("imgBefore"), imgAfter = $("imgAfter");
function setMsg(text) {
  $("msgBefore").textContent = text; $("msgAfter").textContent = text;
}
function syncFeeds() {
  const needBefore = state.view !== "full", needAfter = state.view !== "plain";
  const on = state.running;
  const key = `${on}|${needBefore}|${needAfter}|${state.sessionId}`;
  if (key === state.feedKey) return;
  state.feedKey = key;
  const t = Date.now();
  const apply = (img, need, view) => {
    if (on && need) { img.src = `/api/v1/stream?view=${view}&t=${t}`; img.hidden = false; }
    else { img.removeAttribute("src"); img.hidden = true; }
  };
  apply(imgBefore, needBefore, "plain"); apply(imgAfter, needAfter, "full");
  if (on) setMsg("");
}
function showEnded(stateName) {
  const msg = { failed: "Source unavailable - processing has stopped.", completed: "Video finished.", stopped: "Processing stopped." }[stateName];
  if (msg) setMsg(msg);
}
function setView(view) {
  state.view = view;
  $("panes").dataset.view = view;
  $("panes").classList.toggle("compare", view === "compare");
  document.querySelectorAll(".seg").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  syncFeeds(); if (zone.editing) drawZone();
}
document.querySelectorAll(".seg").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

/* ====================== gauge ====================== */
function updateGauge(objects) {
  const top = objects.reduce((best, o) => (o.score != null && (!best || o.score > best.score) ? o : best), null);
  const score = top ? top.score : 0;
  $("needle").style.transform = `rotate(${(Math.max(0, Math.min(100, score)) - 50) * 1.8}deg)`;
  $("gaugeScore").textContent = top ? score.toFixed(0) : "0";
  const lv = $("gaugeLevel");
  lv.textContent = top ? top.level : "NO ACTIVITY";
  lv.className = `gauge-level lvl-${top ? top.level : "none"}`;
  const who = $("gaugeWho");
  if (!top) { who.textContent = state.running ? "Watching... nothing tracked yet." : "Highest-risk object appears here."; return; }
  const tags = (top.behaviors || []).map((b) => b.label.toLowerCase()).join(", ");
  who.textContent = `#${top.track_id ?? "-"} ${top.class_name}${tags ? " - " + tags : ""}`;
}

/* ====================== pipeline strip ====================== */
function updatePipeline(s) {
  const objs = s.objects || [];
  if (s.session_id !== state.seenSession) { state.seenSession = s.session_id; state.seenIds = new Set(); state.alertCount = 0; }
  objs.forEach((o) => o.track_id != null && state.seenIds.add(o.track_id));
  const inCtx = objs.filter((o) => o.zone_status === "inside" || o.zone_status === "near").length;
  const tags = objs.reduce((n, o) => n + (o.behaviors || []).length, 0);
  const top = objs.reduce((m, o) => Math.max(m, o.score ?? 0), 0);
  const set = (id, val, active, alarm) => {
    $(id).textContent = val;
    const step = $(id).closest(".step");
    step.classList.toggle("active", !!active); step.classList.toggle("alarm", !!alarm);
  };
  set("pDetect", objs.length, objs.length > 0);
  set("pTrack", state.seenIds.size, state.seenIds.size > 0);
  set("pContext", inCtx, inCtx > 0);
  set("pBehavior", tags, tags > 0);
  set("pScore", objs.length ? top.toFixed(0) : "-", objs.length > 0, top > 60);
  set("pAlert", state.alertCount, state.alertCount > 0, objs.some((o) => o.level === "HIGH"));
}

/* ====================== objects + explanation ====================== */
function renderTelemetry(objects) {
  const body = $("telemetryBody");
  body.replaceChildren();
  $("objInfo").textContent = objects.length ? `${objects.length} object(s) in view` : "";
  if (!objects.length) { body.append(el("tr", {}, el("td", { colspan: 11, class: "empty", text: state.running ? "Nothing tracked on this frame" : "No objects in view" }))); return; }
  [...objects].sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).forEach((o) => {
    const zone = o.zone_name ? `${o.zone_name} (${o.zone_status})` : "-";
    const tr = el("tr", { class: o.track_id === state.selected ? "sel" : "" },
      el("td", { text: o.track_id != null ? `#${o.track_id}` : "-" }), el("td", { text: o.class_name }),
      el("td", { text: `${Math.round(o.confidence * 100)}%` }), el("td", { text: `${o.speed_px_s.toFixed(0)} px/s` }),
      el("td", { text: o.direction }), el("td", { text: o.approach }), el("td", { text: zone }),
      el("td", { text: o.dwell_s != null ? `${o.dwell_s.toFixed(1)} s` : "-" }),
      el("td", { class: "num", text: fmt1(o.score) }), el("td", {}, pill(o.level)),
      el("td", {}, el("div", { class: "chips" }, ...(o.behaviors || []).map((b) => chip(b.code)))));
    tr.addEventListener("click", () => { state.selected = o.track_id; renderObjects(state.objects); });
    body.append(tr);
  });
}

function renderObjects(objects) {
  state.objects = objects;
  const list = $("objList");
  const sorted = [...objects].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  if (!sorted.some((o) => o.track_id === state.selected)) state.selected = sorted[0]?.track_id ?? null;
  renderTelemetry(objects);
  list.replaceChildren();
  if (!sorted.length) { list.append(el("p", { class: "empty", text: "No objects in view." })); renderExplain(null); return; }
  sorted.forEach((o) => {
    const tags = (o.behaviors || []).map((b) => b.label).join(" - ");
    const zone = o.zone_name ? `${o.zone_name}: ${o.zone_status}` : "no zone";
    const btn = el("button", { class: `obj${o.track_id === state.selected ? " sel" : ""}`, role: "option", type: "button" },
      el("b", { text: `#${o.track_id ?? "-"} ${o.class_name}` }),
      el("span", { class: "score" }, pill(o.level), document.createTextNode(` ${fmt1(o.score)}`)),
      el("span", { class: "sub", text: `${zone}${tags ? " - " + tags : ""}` }));
    btn.addEventListener("click", () => { state.selected = o.track_id; renderObjects(state.objects); });
    list.append(btn);
  });
  renderExplain(sorted.find((o) => o.track_id === state.selected) || null);
}

function renderExplain(o) {
  const body = $("explainBody");
  body.replaceChildren();
  if (!o) { body.append(el("p", { class: "empty", text: "Select an object to see how its score was built from five weighted factors and what it is doing." })); return; }
  const factors = Object.values(o.factors || {});
  const known = factors.filter((f) => f.known).sort((a, b) => b.contribution - a.contribution);
  const parts = known.filter((f) => f.contribution > 0.5).slice(0, 2).map((f) => `${f.detail} (+${f.contribution.toFixed(1)})`);
  const sentence = parts.length ? `Scored ${o.score.toFixed(1)}/100 mainly because: ${parts.join("; and ")}.`
    : `Scored ${o.score.toFixed(1)}/100 - no factor contributes strongly.`;
  body.append(el("h3", {}, document.createTextNode(`${o.class_name} #${o.track_id ?? "-"}  `), pill(o.level),
    document.createTextNode(`  ${fmt1(o.score)} / 100`)));
  body.append(el("p", { class: "sentence", text: sentence }));
  const zoneTxt = o.zone_name ? `${o.zone_status} ${o.zone_name}` : "no zone";
  body.append(el("p", { class: "measure", text: `Measured: confidence ${Math.round(o.confidence * 100)}% - speed ${o.speed_px_s.toFixed(0)} px/s - moving ${o.direction} - approach ${o.approach} - ${zoneTxt} - dwell ${o.dwell_s != null ? o.dwell_s.toFixed(1) + " s" : "n/a"}` }));
  if ((o.behaviors || []).length) {
    const row = el("div", { class: "chips" });
    o.behaviors.forEach((b) => { const c = chip(b.code); c.title = b.detail; c.textContent = `${b.label} - ${b.detail}`; row.append(c); });
    body.append(row);
  }
  const wrap = el("div", { class: "factors" });
  factors.forEach((f) => {
    const row = el("div", { class: `factor${f.known ? "" : " unknown"}` },
      el("span", { class: "name" }, document.createTextNode(FACTOR_LABEL[f.name] || f.name), helpIcon(FACTOR_TERM[f.name])),
      el("div", { class: "bar", title: `value ${f.value} x weight ${f.weight}` }, el("i", { style: `width:${Math.round(f.value * 100)}%` })),
      el("span", { class: "pts", text: `+${f.contribution.toFixed(1)}` }),
      el("span", { class: "detail", text: `${f.known ? "" : "unknown - "}${f.detail} (weight ${Math.round(f.weight * 100)}%)` }));
    wrap.append(row);
  });
  body.append(wrap);
}

function renderFormula() {
  const w = state.config?.weights; if (!w) return;
  $("formulaLine").textContent =
    `Score = 100 x (${w.object_risk}*ObjectRisk + ${w.confidence}*Confidence + ${w.movement}*Movement + ${w.proximity}*Proximity + ${w.persistence}*Persistence)` +
    `   |   LOW 0-30 - MEDIUM 31-60 - HIGH 61-100   |   weights are configurable defaults, not validated probabilities`;
}

/* ====================== alerts ====================== */
function topFactors(ev) {
  return Object.values(ev.factors || {}).filter((f) => f.contribution >= 1).sort((a, b) => b.contribution - a.contribution)
    .slice(0, 3).map((f) => `${FACTOR_LABEL[f.name] || f.name} +${f.contribution.toFixed(1)}`);
}
function alertCard(ev) {
  const codes = (ev.movement?.behaviors) || [];
  const card = el("div", { class: `alert ${ev.level}`, tabindex: "0" },
    el("div", { class: "alert-top" }, pill(ev.level),
      el("span", { text: ev.event_type === "zone_entry" ? "Zone entry" : "Level change" }),
      el("span", { text: `#${ev.track_id} ${ev.class_name}` }), el("span", { class: "when", text: fmtTime(ev.timestamp) })),
    el("div", { class: "alert-title", text: `${ev.reason}` }),
    el("div", { class: "chips" }, ...codes.map(chip)),
    el("div", { class: "alert-detail", text: `${ev.zone_name ? "Zone: " + ev.zone_name + ". " : ""}Score ${ev.score}/100. ${topFactors(ev).join(" - ")}. ${ev.explanation || ""}` }));
  const toggle = () => card.classList.toggle("open");
  card.addEventListener("click", toggle);
  card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
  return card;
}
function renderAlerts() {
  const feed = $("alertFeed");
  feed.replaceChildren();
  if (!state.alerts.length) { feed.append(el("p", { class: "empty", text: "No alerts yet. An alert fires when an object enters a zone or its threat level changes." })); return; }
  // HIGH first, then newest first
  [...state.alerts].sort((a, b) => (b.priority - a.priority) || (b._seq - a._seq)).forEach((ev) => feed.append(alertCard(ev)));
}
let alertSeq = 0;
function onLiveEvent(ev) {
  ev._seq = alertSeq++;
  state.alerts.unshift(ev); state.alerts.length = Math.min(state.alerts.length, MAX_ALERTS);
  state.alertCount += 1;
  renderAlerts();
  const filter = $("levelFilter").value;
  if (state.page === 1 && (!filter || filter === ev.level)) { prependEventRow(ev); }
  refreshChartsSoon();
}

/* ====================== event history ====================== */
function eventRow(ev, highlight) {
  const codes = ev.movement?.behaviors || [];
  const tr = el("tr", highlight && ev.level === "HIGH" ? { class: "new-high" } : {},
    el("td", { text: fmtDateTime(ev.timestamp) }), el("td", {}, pill(ev.level)), el("td", { text: fmt1(ev.score) }),
    el("td", { text: `#${ev.track_id} ${ev.class_name}` }), el("td", { text: ev.zone_name || "-" }),
    el("td", {}, el("div", { class: "chips" }, ...codes.map(chip))),
    el("td", { class: "reason", text: ev.reason }));
  tr.style.cursor = "pointer";
  tr.title = "Click to show the full score explanation";
  tr.addEventListener("click", () => {
    const next = tr.nextElementSibling;
    if (next && next.classList.contains("detail")) { next.remove(); return; }
    tr.after(el("tr", { class: "detail" }, el("td", { colspan: 7, class: "reason", text: ev.explanation })));
  });
  return tr;
}
function prependEventRow(ev) {
  const body = $("eventsBody");
  if (body.querySelector("td.empty")) body.replaceChildren();
  body.prepend(eventRow(ev, true));
  const rows = [...body.children].filter((r) => !r.classList.contains("detail"));
  while (rows.length > PAGE_SIZE) rows.pop().remove();
}
async function loadEvents() {
  const q = new URLSearchParams({ page: state.page, page_size: PAGE_SIZE });
  const level = $("levelFilter").value; if (level) q.set("level", level);
  try {
    const data = await api(`/api/v1/events?${q}`);
    const body = $("eventsBody"); body.replaceChildren();
    if (!data.items.length) body.append(el("tr", {}, el("td", { colspan: 7, class: "empty", text: "No events" })));
    data.items.forEach((ev) => body.append(eventRow(ev, false)));
    const pages = Math.max(1, Math.ceil(data.total / PAGE_SIZE));
    $("pageInfo").textContent = `Page ${state.page} of ${pages} (${data.total} events)`;
    $("prevBtn").disabled = state.page <= 1; $("nextBtn").disabled = state.page >= pages;
  } catch (e) { showError(e.message); }
}

/* ====================== charts ====================== */
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let levelChart, timelineChart, chartTimer;
function initCharts() {
  if (typeof Chart === "undefined") return;
  Chart.defaults.color = cssVar("--muted"); Chart.defaults.borderColor = cssVar("--border");
  const colors = [cssVar("--low"), cssVar("--medium"), cssVar("--high")];
  levelChart = new Chart($("levelChart"), { type: "doughnut",
    data: { labels: ["LOW", "MEDIUM", "HIGH"], datasets: [{ data: [0, 0, 0], backgroundColor: colors, borderWidth: 0 }] },
    options: { maintainAspectRatio: false, cutout: "62%", plugins: { legend: { position: "bottom" } } } });
  timelineChart = new Chart($("timelineChart"), { type: "bar",
    data: { labels: [], datasets: ["LOW", "MEDIUM", "HIGH"].map((l, i) => ({ label: l, data: [], backgroundColor: colors[i] })) },
    options: { maintainAspectRatio: false, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } } },
      plugins: { legend: { display: false } } } });
}
async function refreshCharts() {
  if (!levelChart) return;
  try {
    const s = await api("/api/v1/analytics/summary");
    levelChart.data.datasets[0].data = [s.by_level.LOW, s.by_level.MEDIUM, s.by_level.HIGH]; levelChart.update();
    const tl = s.timeline.slice(-24);
    timelineChart.data.labels = tl.map((b) => new Date(b.bucket).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    ["LOW", "MEDIUM", "HIGH"].forEach((l, i) => { timelineChart.data.datasets[i].data = tl.map((b) => b[l]); });
    timelineChart.update();
  } catch (e) { showError(e.message); }
}
function refreshChartsSoon() { clearTimeout(chartTimer); chartTimer = setTimeout(refreshCharts, 800); }

/* ====================== status from the server ====================== */
function applyStatus(s) {
  const wasRunning = state.running;
  state.running = s.running; state.sessionId = s.session_id ?? state.sessionId;
  const cls = s.state === "running" ? "badge-on" : s.state === "failed" ? "badge-bad" : "badge-off";
  if (s.model === "loading" && !s.running) setBadge($("stateBadge"), "Loading model...", "badge-warn");
  else setBadge($("stateBadge"), STATE_LABEL[s.state] || s.state, cls);
  $("procInfo").textContent = s.running ? `${s.source} - ${s.fps} fps - ${s.frames_processed} frames`
    : (s.source ? `${s.source} - ${STATE_LABEL[s.state]}` : "No active source");
  $("sessionInfo").textContent = s.session_id ? `Session #${s.session_id}` : "";
  $("cntLow").textContent = s.object_levels.LOW; $("cntMed").textContent = s.object_levels.MEDIUM; $("cntHigh").textContent = s.object_levels.HIGH;
  $("startBtn").disabled = s.running; $("stopBtn").disabled = !s.running;
  $("quickDemoBtn").disabled = s.running;
  $("recBadge").hidden = !(s.running && s.recording);
  if (s.state === "failed" && s.error) showError(`Processing error: ${s.error}`); else if (s.running) showError("");
  if (!modeIsImage() || s.running) { updatePipeline(s); updateGauge(s.objects || []); renderObjects(s.objects || []); }
  updatePhoneState(s);
  syncFeeds();
  if (!s.running && (wasRunning || s.state !== "idle")) showEnded(s.state);
  if (wasRunning && !s.running) { loadEvents(); refreshCharts(); loadRecordings(); if (zone.editing) closeZoneEditor(); }
}

/* ====================== zone editor ====================== */
const zone = { editing: false, pts: [] };
const canvas = $("zoneCanvas");
function contentRect() {
  const box = canvas.getBoundingClientRect();
  const nw = imgAfter.naturalWidth || 640, nh = imgAfter.naturalHeight || 480;
  const k = Math.min(box.width / nw, box.height / nh), w = nw * k, h = nh * k;
  return { box, w, h, x0: (box.width - w) / 2, y0: (box.height - h) / 2 };
}
function drawZone() {
  const { box, w, h, x0, y0 } = contentRect();
  canvas.width = box.width; canvas.height = box.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!zone.pts.length) return;
  const P = zone.pts.map((p) => [x0 + p.x * w, y0 + p.y * h]);
  ctx.lineWidth = 2; ctx.strokeStyle = "#2dd4bf"; ctx.fillStyle = "rgba(45,212,191,.22)";
  ctx.beginPath(); P.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  if (P.length >= 3) { ctx.closePath(); ctx.fill(); }
  ctx.stroke();
  P.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fillStyle = "#fff"; ctx.fill(); ctx.stroke(); });
}
function openZoneEditor() {
  if (!state.running) { showError("Start a source first, then draw the zone on the live VisionShield view."); return; }
  if (state.view === "plain") setView("compare");
  zone.editing = true; zone.pts = [];
  $("zoneBar").hidden = false; canvas.hidden = false; $("zoneBtn").classList.add("on");
  drawZone();
}
function closeZoneEditor() {
  zone.editing = false; zone.pts = [];
  $("zoneBar").hidden = true; canvas.hidden = true; $("zoneBtn").classList.remove("on");
}
canvas.addEventListener("click", (e) => {
  const { box, w, h, x0, y0 } = contentRect();
  const nx = (e.clientX - box.left - x0) / w, ny = (e.clientY - box.top - y0) / h;
  if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;
  zone.pts.push({ x: nx, y: ny }); drawZone();
});
window.addEventListener("resize", () => zone.editing && drawZone());
async function saveZones(zones) {
  try {
    const [rw, rh] = state.zones.reference_size;
    state.zones = await api("/api/v1/zones", jsonOpts("PUT", { reference_size: [rw, rh], zones }));
    showError("");
  } catch (e) { showError(e.message); }
}
$("zoneBtn").addEventListener("click", () => (zone.editing ? closeZoneEditor() : openZoneEditor()));
$("zoneCancel").addEventListener("click", closeZoneEditor);
$("zoneUndo").addEventListener("click", () => { zone.pts.pop(); drawZone(); });
$("zoneSave").addEventListener("click", async () => {
  if (zone.pts.length < 3) return showError("A zone needs at least 3 points.");
  const [rw, rh] = state.zones.reference_size;
  const name = $("zoneName").value.trim() || "Restricted Area";
  const fresh = { id: `zone-${Date.now().toString(36)}`, name, restricted: true,
    polygon: zone.pts.map((p) => [Math.round(p.x * rw * 10) / 10, Math.round(p.y * rh * 10) / 10]) };
  await saveZones([...state.zones.zones, fresh]);
  zone.pts = []; drawZone();
});
$("zoneClear").addEventListener("click", async () => { await saveZones([]); zone.pts = []; drawZone(); });
$("zoneReset").addEventListener("click", async () => {
  try { state.zones = await api("/api/v1/zones/reset", { method: "POST" }); showError(""); } catch (e) { showError(e.message); }
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && zone.editing) closeZoneEditor(); });

/* ====================== phone / browser camera ====================== */
async function loadNetwork() {
  try { state.network = await api("/api/v1/network"); } catch (_) { state.network = null; }
  const box = $("qrBox"); box.replaceChildren();
  const urls = state.network?.phone_urls || [];
  if (urls.length && typeof QRCode !== "undefined") {
    new QRCode(box, { text: urls[0], width: 128, height: 128 });
    $("phoneUrl").textContent = urls[0]; $("phoneHint").textContent = urls.length > 1 ? `Other addresses: ${urls.slice(1).join("  ")}` : "";
  } else if (urls.length) {
    $("phoneUrl").textContent = urls[0]; box.textContent = "QR library unavailable";
  } else {
    $("phoneUrl").textContent = "(HTTPS not enabled)";
    box.textContent = "HTTPS off";
    $("phoneHint").textContent = state.network?.hint || "Run `python scripts/run_demo.py` to enable phone access.";
  }
}
function updatePhoneState(s) {
  const out = $("phoneState");
  if (s.running && s.source_kind === "browser") {
    const live = s.browser_frames > 0;
    out.textContent = live ? `Camera connected - ${s.browser_frames} frames received` : "Waiting for the phone to connect...";
    out.className = `phone-state${live ? " ok" : ""}`;
  } else { out.textContent = ""; }
}

/* ====================== settings explainer ====================== */
function kvTable(rows) {
  const t = el("table", { class: "kv" });
  rows.forEach(([k, v, note]) => t.append(el("tr", {}, el("th", { text: k }), el("td", { text: v }), el("td", { class: "muted", text: note || "" }))));
  return t;
}
function renderSettings() {
  const c = state.config, box = $("settingsBody");
  if (!c) return;
  box.replaceChildren();
  const w = c.weights, pct = (x) => `${Math.round(x * 100)}%`;
  const risks = Object.entries(c.object_risk).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" - ");
  const rows = [
    ["Object risk", w.object_risk, "How much attention this kind of object deserves", `Per-class table: ${risks}; other classes ${c.default_object_risk.toFixed(2)}. Raised by context (unattended item 0.85, crowding +0.15).`],
    ["Detector confidence", w.confidence, "How certain the detector is", "Used as reported (0-1)."],
    ["Movement", w.movement, "How fast it moves and whether it approaches a restricted zone", `Speed >= ${c.speed_ref_diag_per_s} x frame diagonal per second counts as full speed (60% of the factor); +40% if approaching a zone.`],
    ["Zone proximity", w.proximity, "How close it is to a restricted zone", `Inside = 1; falls linearly to 0 at ${pct(c.zone_near_fraction)} of the frame diagonal.`],
    ["Persistence", w.persistence, "How long it has stayed in a monitored zone", `Dwell time / ${c.dwell_ref_s} s, capped at 1.`],
  ];
  const factors = el("table", { class: "kv factors-table" },
    el("thead", {}, el("tr", {}, ...["Factor", "Weight", "What it measures", "How it becomes 0-1"].map((h) => el("th", { text: h })))),
    el("tbody", {}, ...rows.map(([n, wt, what, how]) => el("tr", {}, el("th", { text: n }), el("td", { class: "num", text: pct(wt) }), el("td", { text: what }), el("td", { class: "muted", text: how })))));
  box.append(el("h4", { text: "How the 0-100 score is built (weights sum to 100%)" }), factors);
  box.append(el("h4", { text: "Levels, alerts and behaviour rules" }), kvTable([
    ["Levels", "LOW 0-30 - MEDIUM 31-60 - HIGH 61-100", "A fraction above 30 or 60 falls into the next level."],
    ["Alert cooldown", `${c.alert_cooldown_s} s`, "Minimum gap before a lower level or a repeat zone entry is reported again."],
    ["Alert minimum level", c.alert_min_level, "Levels below this do not raise alerts (they are still scored and drawn)."],
    ["Loitering after", `${c.behavior.loiter_s} s`, "Stationary inside a restricted zone."],
    ["Unattended item after", `${c.behavior.unattended_s} s`, "Backpack/suitcase/handbag stationary with nobody nearby."],
    ["Crowding at", `${c.behavior.crowd_count} people`, "In the same restricted zone."],
    ["Moving fast at", `${c.behavior.run_speed_diag_per_s} x frame diagonal / s`, "For people."],
  ]));
  box.append(el("h4", { text: "Detection, tracking and recording" }), kvTable([
    ["Detector model", c.yolo_model, "Pretrained general-purpose model (not trained for security)."],
    ["Detection confidence cut-off", `${c.detection.confidence}`, "Detections below this are ignored."],
    ["Tracker", c.tracker_config, "ByteTrack keeps IDs stable across frames."],
    ["Processing rate cap", `${c.max_fps} fps`, "Upper limit on frames analysed per second."],
    ["Recording", `${c.recording.enabled_by_default ? "on by default" : "off by default"} - ${c.recording.fps} fps - keeps last ${c.recording.keep_last}`, "Annotated video of each session."],
  ]));
}

/* ====================== recordings ====================== */
const fmtSize = (b) => (b > 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);
async function loadRecordings() {
  let items = [];
  try { items = await api("/api/v1/recordings"); } catch (e) { showError(e.message); return; }
  const list = $("recList"); list.replaceChildren();
  if (!items.length) { list.append(el("p", { class: "empty", text: "No recordings yet. Tick \"Record session\" before pressing Start." })); return; }
  items.forEach((r) => {
    const actions = el("div", { class: "rec-actions" });
    const row = el("div", { class: "rec-row" },
      el("div", { class: "rec-info" }, el("b", { text: `Session #${r.session_id}` }),
        el("span", { class: "muted", text: `${r.source_label} - ${fmtDateTime(r.started_at)} - ${r.seconds != null ? r.seconds.toFixed(0) + " s" : "?"} - ${fmtSize(r.size_bytes)}` })),
      actions);
    const play = el("button", { class: "btn btn-sm btn-primary", type: "button", text: "Replay" });
    play.addEventListener("click", () => {
      $("replayTitle").textContent = `Replay - session #${r.session_id}`;
      $("replayImg").src = `/api/v1/recordings/${r.session_id}/replay?t=${Date.now()}`;
      $("replayBox").hidden = false; $("replayBox").scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
    const dl = el("a", { class: "btn btn-sm", href: `/api/v1/recordings/${r.session_id}/download`, download: "", text: "Download" });
    const del = el("button", { class: "btn btn-sm", type: "button", text: "Delete" });
    del.addEventListener("click", async () => {
      if (!confirm(`Delete the recording of session #${r.session_id}?`)) return;
      try { await api(`/api/v1/recordings/${r.session_id}`, { method: "DELETE" }); } catch (e) { showError(e.message); }
      loadRecordings();
    });
    actions.append(play, dl, del); list.append(row);
  });
}
$("replayClose").addEventListener("click", () => { $("replayImg").removeAttribute("src"); $("replayBox").hidden = true; });
$("recRefresh").addEventListener("click", loadRecordings);

/* ====================== controls ====================== */
const modeIsImage = () => $("modeSelect").value === "image";
function syncMode() {
  const mode = $("modeSelect").value;
  $("videoPick").hidden = mode !== "video"; $("videoUpload").hidden = mode !== "video";
  $("imagePick").hidden = mode !== "image";
  $("phonePanel").hidden = mode !== "browser";
  $("startBtn").textContent = mode === "image" ? "Analyse" : "Start";
  $("recordPick").hidden = mode === "image";
  $("sourceHint").textContent = SOURCE_HINT[mode] || "";
  if (mode === "browser" && !state.network) loadNetwork();
}
async function loadSources() {
  try {
    const src = await api("/api/v1/sources");
    const sel = $("videoSelect"); sel.replaceChildren();
    if (!src.videos.length) sel.append(el("option", { value: "", text: "(no videos in data/videos)" }));
    src.videos.forEach((v) => sel.append(el("option", { value: v, text: v })));
    [...$("modeSelect").options].find((o) => o.value === "stream").disabled = !src.stream_configured;
  } catch (e) { showError(e.message); }
}
async function loadConfigAndZones() {
  try {
    state.config = await api("/api/v1/config"); renderFormula(); renderSettings();
    $("recordChk").checked = !!state.config.recording?.enabled_by_default;
  } catch (_) { /* optional */ }
  try { state.zones = await api("/api/v1/zones"); } catch (_) { /* optional */ }
}
async function start() {
  showError("");
  const mode = $("modeSelect").value;
  if (mode === "image") return analyseImage();
  const body = { source_type: mode, record: $("recordChk").checked };
  if (mode === "video") {
    body.video_name = $("videoSelect").value;
    if (!body.video_name) return showError("Put a video in data/videos or upload one first.");
  }
  $("startBtn").disabled = true;
  try {
    const s = await api("/api/v1/sessions", jsonOpts("POST", body));
    state.sessionId = s.id; state.running = true; state.feedKey = ""; state.alerts = []; renderAlerts();
    setMsg(mode === "browser" ? "Waiting for the phone camera..." : "Loading model and opening source...");
    syncFeeds();
  } catch (e) { showError(e.message); $("startBtn").disabled = false; }
}
async function stop() {
  if (state.sessionId == null) return;
  $("stopBtn").disabled = true;
  try { await api(`/api/v1/sessions/${state.sessionId}/stop`, { method: "POST" }); } catch (e) { showError(e.message); }
  await loadEvents(); refreshCharts(); setTimeout(loadRecordings, 600);
}
async function quickDemo() {
  if (state.running) return;
  const names = [...$("videoSelect").options].map((o) => o.value).filter(Boolean);
  if (!names.length) return showError("No videos in data/videos. Upload one or run scripts/make_demo_video.py.");
  $("modeSelect").value = "video"; syncMode();
  $("videoSelect").value = names.find((n) => /people|store|person/i.test(n)) || names[0];
  setView("compare"); start();
}
async function analyseImage() {
  const file = $("imageFile").files[0];
  if (!file) return showError("Choose an image first.");
  const form = new FormData(); form.append("file", file);
  $("startBtn").disabled = true;
  try {
    const r = await api("/api/v1/images/analyze", { method: "POST", body: form });
    setView("full");
    imgAfter.src = r.annotated_image; imgAfter.hidden = false; setMsg("");
    renderObjects(r.objects); updateGauge(r.objects);
    $("procInfo").textContent = `Image ${r.width}x${r.height} - ${r.object_count} object(s)${r.max_level ? " - max " + r.max_level : ""}`;
    $("objInfo").textContent += ` - ${r.note}`;
  } catch (e) { showError(e.message); } finally { $("startBtn").disabled = state.running; }
}
async function uploadVideo() {
  const file = $("videoFile").files[0]; if (!file) return;
  const form = new FormData(); form.append("file", file);
  try { const r = await api("/api/v1/videos/upload", { method: "POST", body: form }); await loadSources(); $("videoSelect").value = r.video_name; }
  catch (e) { showError(e.message); } finally { $("videoFile").value = ""; }
}

$("modeSelect").addEventListener("change", syncMode);
$("startBtn").addEventListener("click", start);
$("stopBtn").addEventListener("click", stop);
$("quickDemoBtn").addEventListener("click", quickDemo);
$("videoFile").addEventListener("change", uploadVideo);
$("imageFile").addEventListener("change", () => { const f = $("imageFile").files[0]; if (f) $("procInfo").textContent = `Selected: ${f.name}`; });
$("refreshBtn").addEventListener("click", () => { loadEvents(); refreshCharts(); });
$("levelFilter").addEventListener("change", () => { state.page = 1; loadEvents(); });
$("prevBtn").addEventListener("click", () => { state.page = Math.max(1, state.page - 1); loadEvents(); });
$("nextBtn").addEventListener("click", () => { state.page += 1; loadEvents(); });

/* ====================== live link ====================== */
function connectWs() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/events`);
  state.ws = ws;
  ws.onopen = () => { state.retry = 0; setBadge($("wsBadge"), "Live link: connected", "badge-on"); };
  ws.onmessage = (m) => {
    let msg; try { msg = JSON.parse(m.data); } catch (_) { return; }
    if (msg.type === "status") applyStatus(msg); else if (msg.type === "event") onLiveEvent(msg.event);
  };
  ws.onclose = () => {
    setBadge($("wsBadge"), "Live link: reconnecting", "badge-warn");
    state.retry = Math.min(state.retry + 1, 6); setTimeout(connectWs, 500 * 2 ** state.retry);
  };
  ws.onerror = () => ws.close();
}

/* ====================== init ====================== */
applyHelp(); renderGlossary($("glossary"));
initCharts(); setView("compare"); syncMode(); setMsg("Select a source and press Start.");
loadSources(); loadConfigAndZones(); loadEvents(); loadRecordings(); refreshCharts(); connectWs();
