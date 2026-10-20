"use strict";

const $ = (id) => document.getElementById(id);
const TARGET_FPS = 8;
const MAX_WIDTH = 640;
const JPEG_QUALITY = 0.7;
const MAX_BUFFERED = 512 * 1024; // skip frames while the socket is backed up

let stream = null, ws = null, timer = null, wakeLock = null, sent = 0;
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

function say(text, bad) { const m = $("msg"); m.textContent = text; m.hidden = !text; m.classList.toggle("bad", !!bad); }
function badge(text, cls) { const b = $("badge"); b.textContent = text; b.className = `badge ${cls || ""}`; }

async function start() {
  say("");
  if (!navigator.mediaDevices?.getUserMedia) {
    return say(window.isSecureContext ? "This browser has no camera API." :
      "Camera access needs HTTPS. Open the https:// address shown on the dashboard (accept the certificate warning).", true);
  }
  $("startBtn").disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: $("facing").value }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    $("startBtn").disabled = false;
    return say(`Could not open the camera: ${e.name}. Allow camera access and try again.`, true);
  }
  $("preview").srcObject = stream;
  wanted = true; attempts = 0;
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch (_) { /* optional */ }
  connect();
}

let wanted = false, retryTimer = null, attempts = 0, waiting = false;

function connect() {
  clearTimeout(retryTimer);
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/camera`);
  ws.binaryType = "arraybuffer";
  const mine = ws;
  ws.onopen = () => badge("Connecting...", "");
  ws.onmessage = (e) => {
    let m; try { m = JSON.parse(e.data); } catch (_) { return; }
    if (m.type === "ready") { attempts = 0; waiting = false; say(""); badge("Streaming", "live"); $("stopBtn").disabled = false; beginSending(); }
    else if (m.type === "error") { waiting = true; badge("Waiting for dashboard...", "err"); say("No session is waiting for this camera yet. On the dashboard choose Phone / browser camera and press Start - this page keeps retrying."); }
    else if (m.type === "stopped") { say("The dashboard stopped the session."); wanted = false; shutdown(false); }
    else if (m.type === "ack") { badge(`Streaming - ${m.frames} frames`, "live"); }
  };
  ws.onclose = () => {
    if (mine !== ws) return;           // an older socket
    clearInterval(timer); timer = null;
    if (!wanted) return;
    attempts += 1;
    if (attempts > 100) { say("Could not reconnect. Press Start camera to try again.", true); return shutdown(false); }
    badge(waiting ? "Waiting for dashboard..." : "Reconnecting...", "err");
    retryTimer = setTimeout(connect, Math.min(3000, 500 * attempts));   // survives Wi-Fi blips and screen sleep
  };
  ws.onerror = () => { /* onclose follows and retries */ };
}

function beginSending() {
  clearInterval(timer);
  timer = setInterval(() => {
    const video = $("preview");
    if (!ws || ws.readyState !== WebSocket.OPEN || ws.bufferedAmount > MAX_BUFFERED || !video.videoWidth) return;
    const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => { if (blob && ws && ws.readyState === WebSocket.OPEN) { ws.send(blob); sent += 1; } }, "image/jpeg", JPEG_QUALITY);
  }, 1000 / TARGET_FPS);
}

function shutdown(closeSocket = true) {
  clearInterval(timer); timer = null; clearTimeout(retryTimer);
  if (closeSocket) wanted = false;
  if (closeSocket && ws && ws.readyState <= 1) ws.close();
  ws = null;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  $("preview").srcObject = null;
  try { wakeLock?.release(); } catch (_) { /* ignore */ }
  wakeLock = null;
  badge("Camera off", ""); $("startBtn").disabled = false; $("stopBtn").disabled = true;
}

$("startBtn").addEventListener("click", start);
$("stopBtn").addEventListener("click", () => shutdown(true));
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && stream && !wakeLock) navigator.wakeLock?.request("screen").then((l) => { wakeLock = l; }).catch(() => {});
});
