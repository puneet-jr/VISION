"use strict";

/* One source of truth for "what does this mean?": feeds the hover help (?) and the Guide section. */
const GLOSSARY = {
  // ---- what the system measures for each object
  track_id: { group: "Per-object measurements", title: "ID (#n)",
    text: "A number the tracker gives one physical object while it stays in view. The same ID on consecutive frames means the same object. It identifies an object within this session only - it is not a person's identity and there is no face recognition." },
  class: { group: "Per-object measurements", title: "Class",
    text: "What the detector thinks the object is (person, car, backpack ...). It comes from a general pretrained model (COCO classes); it was not trained for security use." },
  confidence: { group: "Per-object measurements", title: "Confidence",
    text: "How sure the detector is about this object's class and box, 0-100%. It is detector certainty, NOT the probability of a threat." },
  speed: { group: "Per-object measurements", title: "Speed (px/s)",
    text: "How fast the centre of the box moves, in image pixels per second of video. It is not metres per second - that would need a calibrated camera. Near objects look faster than far ones." },
  direction: { group: "Per-object measurements", title: "Direction",
    text: "Where the object is heading on screen (up = towards the top of the image). 'stationary' means it barely moved over the last few seconds." },
  approach: { group: "Per-object measurements", title: "Approach",
    text: "Whether the object has been getting closer to the nearest restricted zone (approaching), moving away (receding), or staying about the same (steady). 'unknown' means not enough observations yet." },
  zone_status: { group: "Per-object measurements", title: "Zone: inside / near / outside",
    text: "Where the box centre is relative to a zone. Inside = within the polygon. Near = outside but within the 'near' distance (a fraction of the frame diagonal, see settings). Outside = further away." },
  dwell: { group: "Per-object measurements", title: "Dwell (s)",
    text: "How long the object has been continuously inside a monitored zone. Short exits (about 2 seconds) do not reset it, so border jitter or a brief occlusion does not restart the clock." },
  behaviors: { group: "Per-object measurements", title: "Behaviours",
    text: "Plain-language tags for what the object is doing, from deterministic rules on its movement, zone and surroundings. See the behaviour definitions below." },

  // ---- score
  score: { group: "Threat score", title: "Threat score (0-100)",
    text: "A weighted sum of five factors (object risk, confidence, movement, zone proximity, persistence) scaled to 0-100. LOW 0-30, MEDIUM 31-60, HIGH 61-100. Weights are configurable engineering defaults, not validated probabilities. A person on their own scores LOW - high scores need a combination of context." },
  object_risk: { group: "Threat score", title: "Object risk (factor)",
    text: "How much attention this kind of object deserves by default (e.g. a knife > a person > a dog), set per class in configuration. Context can raise it: an unattended bag or a crowd counts as higher risk. It never lowers it." },
  f_confidence: { group: "Threat score", title: "Detector confidence (factor)",
    text: "The detector's confidence, used so that uncertain detections count for less." },
  f_movement: { group: "Threat score", title: "Movement (factor)",
    text: "Speed relative to the frame size (saturating at the 'speed reference'), plus extra if the object is approaching a restricted zone. Unknown (shown hatched) when there is no motion history - for example in a single image." },
  f_proximity: { group: "Threat score", title: "Zone proximity (factor)",
    text: "1.0 when inside a restricted zone, falling steadily to 0 at the 'near' distance. 0 if no restricted zone exists." },
  f_persistence: { group: "Threat score", title: "Persistence (factor)",
    text: "Dwell time in a monitored zone divided by the 'dwell reference' (capped at 1). The longer an object stays, the higher it goes." },

  // ---- behaviours
  b_intrusion: { group: "Behaviours", title: "Inside restricted zone", text: "The object's centre is inside a zone marked restricted." },
  b_loitering: { group: "Behaviours", title: "Loitering", text: "Stationary inside a restricted zone for longer than the loiter time (default 6 s)." },
  b_running: { group: "Behaviours", title: "Moving fast", text: "A person moving faster than the run-speed threshold, measured relative to the frame size." },
  b_approaching: { group: "Behaviours", title: "Approaching zone", text: "Close to a restricted zone and getting closer." },
  b_unattended: { group: "Behaviours", title: "Unattended item", text: "A backpack / suitcase / handbag that has been stationary for longer than the unattended time (default 8 s) with no person nearby. Treated as higher risk." },
  b_crowding: { group: "Behaviours", title: "Crowding", text: "Several people (default 3) inside the same restricted zone at once. Raises each person's object risk." },

  // ---- zones, alerts, recording
  zone: { group: "Zones, alerts & recording", title: "Zone",
    text: "A polygon on the image. Restricted zones drive proximity, approach and intrusion; draw or change them with Edit zone and they apply on the next frame." },
  alert: { group: "Zones, alerts & recording", title: "Alert",
    text: "A stored event raised when an object enters a zone or its threat level changes. Escalations alert at once; repeats and flicker are suppressed by a cooldown, and levels below the minimum (default MEDIUM) stay silent. HIGH alerts are listed first." },
  recording: { group: "Zones, alerts & recording", title: "Recording",
    text: "The annotated VisionShield view saved as an .avi video, replayable here or downloadable (opens in VLC / Windows players). Only the most recent recordings are kept." },
  source: { group: "Zones, alerts & recording", title: "Source",
    text: "Recorded video: a file from data/videos. Phone / browser camera: a phone or laptop camera streamed from a web page. Server webcam: a camera plugged into this computer. IP camera: the stream URL set in .env. Upload image: one still image (no movement or persistence)." },

  // ---- pipeline strip
  step_detect: { group: "Pipeline", title: "1 Detect", text: "Objects found on the current frame by the detector." },
  step_track: { group: "Pipeline", title: "2 Track", text: "Distinct IDs seen so far in this session (the same object keeps its ID)." },
  step_context: { group: "Pipeline", title: "3 Context", text: "Objects currently inside or near a zone." },
  step_behavior: { group: "Pipeline", title: "4 Behaviour", text: "Behaviour tags currently active across all objects." },
  step_score: { group: "Pipeline", title: "5 Score", text: "The highest threat score among objects in view." },
  step_alert: { group: "Pipeline", title: "6 Alert", text: "Alerts raised in this session." },
};

function applyHelp() {
  document.querySelectorAll(".help[data-term]").forEach((node) => {
    const g = GLOSSARY[node.dataset.term];
    if (!g) return;
    node.textContent = "?";
    node.dataset.tip = `${g.title}: ${g.text}`;
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "note");
    node.setAttribute("aria-label", `${g.title}: ${g.text}`);
  });
  document.querySelectorAll(".step[data-term]").forEach((node) => {
    const g = GLOSSARY[node.dataset.term];
    if (g) { node.dataset.tip = g.text; node.setAttribute("title", g.text); }
  });
}

function renderGlossary(container) {
  const groups = {};
  Object.entries(GLOSSARY).forEach(([key, g]) => { if (g.group !== "Pipeline") (groups[g.group] ||= []).push(g); });
  container.replaceChildren();
  Object.entries(groups).forEach(([name, items]) => {
    const section = document.createElement("div");
    section.className = "gloss-group";
    const h = document.createElement("h4"); h.textContent = name; section.append(h);
    items.forEach((g) => {
      const row = document.createElement("div"); row.className = "gloss-item";
      const t = document.createElement("b"); t.textContent = g.title;
      const p = document.createElement("p"); p.textContent = g.text;
      row.append(t, p); section.append(row);
    });
    container.append(section);
  });
}
