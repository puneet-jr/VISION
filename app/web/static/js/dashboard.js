"use strict";

const $ = (id) => document.getElementById(id);
const PAGE_SIZE = 20;
const state = { sessionId: null, running: false, page: 1, total: 0, ws: null, retry: 0 };

/* ---------- small helpers (all dynamic text goes through textContent) ---------- */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.entries(props).forEach(([k, v]) => {
    if (k === "class") node.className = v; else if (k === "text") node.textContent = v; else node.setAttribute(k, v);
  });
  children.forEach((c) => c != null && node.append(c));
  return node;
}
function cell(content, cls) {
  const td = el("td", cls ? { class: cls } : {});
  td.append(content instanceof Node ? content : document.createTextNode(content ?? "-"));
  return td;
}
function levelPill(level) { return level ? el("span", { class: `lvl ${level}`, text: level }) : "-"; }
function fmtTime(iso) { const d = new Date(iso); return isNaN(d) ? "-" : d.toLocaleString(); }
function showError(msg) { const b = $("errorBanner"); b.textContent = msg; b.hidden = !msg; }
function setBadge(node, text, cls) { node.textContent = text; node.className = `badge ${cls}`; }

async function api(path, opts = {}) {
  let res;
  try { res = await fetch(path, opts); } catch (e) { throw new Error("Cannot reach the server."); }
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-JSON body */ }
  if (!res.ok) {
    const msg = body?.error?.message || body?.detail || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return body;
}

/* ---------- charts ---------- */
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let levelChart, timelineChart;
function initCharts() {
  if (typeof Chart === "undefined") { return; }
  Chart.defaults.color = cssVar("--muted");
  Chart.defaults.borderColor = cssVar("--border");
  const colors = [cssVar("--low"), cssVar("--medium"), cssVar("--high")];
  levelChart = new Chart($("levelChart"), {
    type: "doughnut",
    data: { labels: ["LOW", "MEDIUM", "HIGH"], datasets: [{ data: [0, 0, 0], backgroundColor: colors, borderWidth: 0 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } },
  });
  timelineChart = new Chart($("timelineChart"), {
    type: "bar",
    data: { labels: [], datasets: ["LOW", "MEDIUM", "HIGH"].map((l, i) => ({ label: l, data: [], backgroundColor: colors[i] })) },
    options: { maintainAspectRatio: false, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } } },
      plugins: { legend: { display: false } } },
  });
}
async function refreshCharts() {
  if (!levelChart) return;
  try {
    const s = await api("/api/v1/analytics/summary");
    levelChart.data.datasets[0].data = [s.by_level.LOW, s.by_level.MEDIUM, s.by_level.HIGH];
    levelChart.update();
    const tl = s.timeline.slice(-24);
    timelineChart.data.labels = tl.map((b) => new Date(b.bucket).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    ["LOW", "MEDIUM", "HIGH"].forEach((l, i) => { timelineChart.data.datasets[i].data = tl.map((b) => b[l]); });
    timelineChart.update();
  } catch (e) { showError(e.message); }
}

/* ---------- telemetry ---------- */
function renderObjects(objects) {
  const body = $("telemetryBody");
  body.replaceChildren();
  $("objInfo").textContent = objects.length ? `${objects.length} object(s)` : "";
  if (!objects.length) { body.append(el("tr", {}, el("td", { colspan: 10, class: "empty", text: "No objects" }))); return; }
  const sorted = [...objects].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  sorted.forEach((o) => {
    const why = o.factors && Object.keys(o.factors).length
      ? Object.values(o.factors).map((f) => `${f.name} ${f.contribution}`).join(" | ") : "-";
    const zone = o.zone_name ? `${o.zone_name} (${o.zone_status})` : "-";
    body.append(el("tr", {},
      cell(o.track_id != null ? `#${o.track_id}` : "-"), cell(o.class_name), cell((o.confidence * 100).toFixed(0) + "%"),
      cell(o.speed_px_s.toFixed(1)), cell(`${o.direction}${o.approach === "approaching" ? " / approaching" : ""}`),
      cell(zone), cell(o.dwell_s != null ? o.dwell_s.toFixed(1) : "-"),
      cell(o.score != null ? o.score.toFixed(1) : "-"), cell(levelPill(o.level)), cell(why, "explain")));
  });
}

/* ---------- events ---------- */
function eventRow(ev, highlight) {
  const m = ev.movement || {};
  const movement = `${m.direction ?? "-"}, ${(m.speed_px_s ?? 0).toFixed(0)} px/s${m.approach && m.approach !== "unknown" ? ", " + m.approach : ""}`;
  const tr = el("tr", highlight && ev.level === "HIGH" ? { class: "new-high" } : {},
    cell(fmtTime(ev.timestamp)), cell(levelPill(ev.level)), cell(Number(ev.score).toFixed(1)),
    cell(`#${ev.track_id} ${ev.class_name}`), cell(ev.zone_name || "-"), cell(movement),
    cell(`${ev.reason}. ${ev.explanation}`, "explain"));
  return tr;
}
async function loadEvents() {
  const level = $("levelFilter").value;
  const q = new URLSearchParams({ page: state.page, page_size: PAGE_SIZE });
  if (level) q.set("level", level);
  try {
    const data = await api(`/api/v1/events?${q}`);
    state.total = data.total;
    const body = $("eventsBody");
    body.replaceChildren();
    if (!data.items.length) body.append(el("tr", {}, el("td", { colspan: 7, class: "empty", text: "No events" })));
    data.items.forEach((ev) => body.append(eventRow(ev, false)));
    const pages = Math.max(1, Math.ceil(data.total / PAGE_SIZE));
    $("pageInfo").textContent = `Page ${state.page} of ${pages} (${data.total} events)`;
    $("prevBtn").disabled = state.page <= 1;
    $("nextBtn").disabled = state.page >= pages;
  } catch (e) { showError(e.message); }
}
function onLiveEvent(ev) {
  const filter = $("levelFilter").value;
  if (state.page === 1 && (!filter || filter === ev.level)) {
    const body = $("eventsBody");
    if (body.querySelector(".empty")) body.replaceChildren();
    body.prepend(eventRow(ev, true));
    while (body.children.length > PAGE_SIZE) body.lastChild.remove();
  }
  refreshChartsSoon();
}
let chartTimer;
function refreshChartsSoon() { clearTimeout(chartTimer); chartTimer = setTimeout(refreshCharts, 800); }

/* ---------- status ---------- */
const STATE_LABEL = { idle: "Idle", running: "Processing", stopped: "Stopped", completed: "Video finished", failed: "Source error" };
function applyStatus(s) {
  state.running = s.running;
  state.sessionId = s.session_id ?? state.sessionId;
  const cls = s.state === "running" ? "badge-on" : s.state === "failed" ? "badge-bad" : "badge-off";
  setBadge($("stateBadge"), STATE_LABEL[s.state] || s.state, cls);
  $("procInfo").textContent = s.running ? `${s.source} - ${s.fps} fps - ${s.frames_processed} frames` : (s.source ? `${s.source} - ${STATE_LABEL[s.state]}` : "No active source");
  $("sessionInfo").textContent = s.session_id ? `Session #${s.session_id}` : "";
  $("cntLow").textContent = s.object_levels.LOW; $("cntMed").textContent = s.object_levels.MEDIUM; $("cntHigh").textContent = s.object_levels.HIGH;
  $("startBtn").disabled = s.running; $("stopBtn").disabled = !s.running;
  if (s.running || s.state !== "idle") renderObjects(s.objects || []);
  if (s.state === "failed" && s.error) showError(`Processing error: ${s.error}`);
  if (!s.running) { showFeed(s.state === "idle" || modeIsImage() ? null : s.state); }
}

/* ---------- video panel ---------- */
let feedOn = false;
function modeIsImage() { return $("modeSelect").value === "image"; }
function showFeed(endedState) {
  const img = $("videoImg"), msg = $("videoMsg");
  if (state.running) {
    if (!feedOn) { img.src = `/api/v1/stream?t=${Date.now()}`; feedOn = true; }
    img.hidden = false; msg.hidden = true; return;
  }
  if (feedOn) { feedOn = false; img.removeAttribute("src"); }
  if (endedState === "failed") { img.hidden = true; msg.hidden = false; msg.textContent = "Source unavailable - processing has stopped."; }
  else if (endedState) { img.hidden = true; msg.hidden = false; msg.textContent = endedState === "completed" ? "Video finished." : "Processing stopped."; }
  else if (!img.src || img.hidden) { img.hidden = true; msg.hidden = false; msg.textContent = modeIsImage() ? "Choose an image to analyse." : "Select a source and press Start."; }
}
$("videoImg").addEventListener("error", () => { if (state.running) showError("Video stream interrupted."); });

/* ---------- websocket ---------- */
function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/events`);
  state.ws = ws;
  ws.onopen = () => { state.retry = 0; setBadge($("wsBadge"), "Live link: connected", "badge-on"); };
  ws.onmessage = (m) => {
    let msg; try { msg = JSON.parse(m.data); } catch (_) { return; }
    if (msg.type === "status") applyStatus(msg); else if (msg.type === "event") onLiveEvent(msg.event);
  };
  ws.onclose = () => {
    setBadge($("wsBadge"), "Live link: reconnecting", "badge-warn");
    state.retry = Math.min(state.retry + 1, 6);
    setTimeout(connectWs, 500 * 2 ** state.retry);
  };
  ws.onerror = () => ws.close();
}

/* ---------- controls ---------- */
function syncMode() {
  const mode = $("modeSelect").value;
  $("videoPick").hidden = mode !== "video"; $("videoUpload").hidden = mode !== "video";
  $("imagePick").hidden = mode !== "image";
  $("startBtn").textContent = mode === "image" ? "Analyse" : "Start";
  if (!state.running) showFeed(null);
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
async function start() {
  showError("");
  const mode = $("modeSelect").value;
  if (mode === "image") return analyseImage();
  const body = { source_type: mode };
  if (mode === "video") {
    body.video_name = $("videoSelect").value;
    if (!body.video_name) return showError("Put a video in data/videos or upload one first.");
  }
  $("startBtn").disabled = true;
  try {
    const s = await api("/api/v1/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    state.sessionId = s.id; state.running = true; showFeed(null);
  } catch (e) { showError(e.message); $("startBtn").disabled = false; }
}
async function stop() {
  if (state.sessionId == null) return;
  $("stopBtn").disabled = true;
  try { await api(`/api/v1/sessions/${state.sessionId}/stop`, { method: "POST" }); } catch (e) { showError(e.message); }
  await loadEvents(); refreshCharts();
}
async function analyseImage() {
  const file = $("imageFile").files[0];
  if (!file) return showError("Choose an image first.");
  const form = new FormData(); form.append("file", file);
  $("startBtn").disabled = true;
  try {
    const r = await api("/api/v1/images/analyze", { method: "POST", body: form });
    const img = $("videoImg"); img.src = r.annotated_image; img.hidden = false; $("videoMsg").hidden = true;
    renderObjects(r.objects);
    $("procInfo").textContent = `Image ${r.width}x${r.height} - ${r.object_count} object(s)${r.max_level ? " - max " + r.max_level : ""}`;
    $("objInfo").textContent += ` - ${r.note}`;
  } catch (e) { showError(e.message); } finally { $("startBtn").disabled = state.running; }
}
async function uploadVideo() {
  const file = $("videoFile").files[0];
  if (!file) return;
  const form = new FormData(); form.append("file", file);
  try { const r = await api("/api/v1/videos/upload", { method: "POST", body: form }); await loadSources(); $("videoSelect").value = r.video_name; }
  catch (e) { showError(e.message); } finally { $("videoFile").value = ""; }
}

$("modeSelect").addEventListener("change", syncMode);
$("startBtn").addEventListener("click", start);
$("stopBtn").addEventListener("click", stop);
$("videoFile").addEventListener("change", uploadVideo);
$("imageFile").addEventListener("change", () => { const f = $("imageFile").files[0]; if (f) $("procInfo").textContent = `Selected: ${f.name}`; });
$("refreshBtn").addEventListener("click", () => { loadEvents(); refreshCharts(); });
$("levelFilter").addEventListener("change", () => { state.page = 1; loadEvents(); });
$("prevBtn").addEventListener("click", () => { state.page = Math.max(1, state.page - 1); loadEvents(); });
$("nextBtn").addEventListener("click", () => { state.page += 1; loadEvents(); });

initCharts(); syncMode(); loadSources(); loadEvents(); refreshCharts(); connectWs();
