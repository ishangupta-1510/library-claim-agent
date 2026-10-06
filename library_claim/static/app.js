// Sweep client: microphone + camera to the server, agent voice and live inventory back.
//
// Two camera streams leave this page:
//   "video"    small JPEG once a second, for the live agent to follow the conversation;
//   "keyframe" full-resolution JPEG for the measuring pipeline, sent only after the
//              previous one was processed (back-pressure), so slow links never queue up.

const $ = (id) => document.getElementById(id);
const LIVE_FRAME_MS = 1000;
const KEYFRAME_MIN_MS = 700;
const KEYFRAME_MAX_SIDE = 1920;

let ws, mediaStream, audioCtx, micNode, playCtx;
let playHead = 0;
let playing = [];
let keyframeInFlight = false;
let lastKeyframeAt = 0;
let timers = [];

function status(text) { $("status").textContent = text; }

function log(speaker, text) {
  const box = $("transcript");
  let last = box.lastElementChild;
  // Streaming transcripts arrive in chunks: extend the current line for the same speaker.
  if (!last || last.dataset.speaker !== speaker || last.dataset.closed) {
    last = document.createElement("p");
    last.dataset.speaker = speaker;
    last.className = speaker === "agent" ? "agent" : "";
    last.textContent = speaker === "agent" ? "Agent: " : "You: ";
    box.appendChild(last);
  }
  last.textContent += text;
  box.scrollTop = box.scrollHeight;
}

function closeTurns() { [...$("transcript").children].forEach((p) => (p.dataset.closed = "1")); }

// ---------- audio in: 16 kHz PCM16 ----------
const WORKLET = `
class Pcm16 extends AudioWorkletProcessor {
  constructor(){ super(); this.buf = []; this.ratio = sampleRate / 16000; }
  process(inputs){
    const ch = inputs[0][0]; if (!ch) return true;
    for (let i = 0; i < ch.length; i += this.ratio) this.buf.push(ch[Math.floor(i)]);
    if (this.buf.length >= 1600) {               // 100 ms
      const out = new Int16Array(this.buf.length);
      for (let i = 0; i < out.length; i++) out[i] = Math.max(-1, Math.min(1, this.buf[i])) * 0x7fff;
      this.port.postMessage(out.buffer, [out.buffer]); this.buf = [];
    }
    return true;
  }
}
registerProcessor("pcm16", Pcm16);`;

function b64(buffer) {
  let s = ""; const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

async function startMic(stream) {
  audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));
  micNode = new AudioWorkletNode(audioCtx, "pcm16");
  micNode.port.onmessage = (e) => send({ type: "audio", data: b64(e.data) });
  audioCtx.createMediaStreamSource(stream).connect(micNode);
}

// ---------- audio out: agent PCM16 (24 kHz) ----------
function playPcm(base64, mime) {
  const rate = Number((/rate=(\d+)/.exec(mime || "") || [])[1] || 24000);
  playCtx ??= new AudioContext({ sampleRate: rate });
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const pcm = new Int16Array(bytes.buffer);
  const buffer = playCtx.createBuffer(1, pcm.length, rate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 0x8000;
  const source = playCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(playCtx.destination);
  playHead = Math.max(playHead, playCtx.currentTime + 0.02);
  source.start(playHead);
  playHead += buffer.duration;
  playing.push(source);
  source.onended = () => (playing = playing.filter((s) => s !== source));
}

function stopPlayback() {
  playing.forEach((s) => { try { s.stop(); } catch {} });
  playing = [];
  playHead = 0;
}

// ---------- camera ----------
// Frames come from the <video> preview by default; AR mode swaps in the WebXR camera.
let grabFrame = (maxSide, quality) => grab(maxSide > 1000 ? $("large") : $("small"), maxSide, quality);

function grab(canvas, maxSide, quality, source = $("preview")) {
  const width = source.videoWidth ?? source.width;
  const height = source.videoHeight ?? source.height;
  if (!width) return null;
  const scale = Math.min(1, maxSide / Math.max(width, height));
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", quality).split(",")[1];
}

function sendLiveFrame() {
  const data = grabFrame(640, 0.6);
  if (data) send({ type: "video", data });
}

let lastDemoFrameSent = -1;

function maybeSendKeyframe() {
  if (keyframeInFlight || performance.now() - lastKeyframeAt < KEYFRAME_MIN_MS) return;
  if (demoFrame >= 0 && demoFrame === lastDemoFrameSent) return; // a held demo picture is one keyframe
  const data = grabFrame(KEYFRAME_MAX_SIDE, 0.88);
  if (!data) return;
  lastDemoFrameSent = demoFrame;
  keyframeInFlight = true;
  lastKeyframeAt = performance.now();
  send({ type: "keyframe", data });
}

// ---------- server events ----------
function send(message) { if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message)); }

function renderBooks(books) {
  $("n-books").textContent = books.length;
  $("n-legible").textContent = books.filter((b) => b.legible).length;
  $("books").innerHTML = books
    .slice()
    .sort((a, b) => (a.shelf + String(a.position).padStart(3)).localeCompare(b.shelf + String(b.position).padStart(3)))
    .map((b) => {
      const reading = b.legible ? `${escapeHtml(b.title)}${b.author ? " · " + escapeHtml(b.author) : ""}` : '<span class="unread">unreadable spine</span>';
      const dims = b.height_cm ? `${b.height_cm} × ${b.thickness_cm}` : "—";
      return `<tr${b.excluded ? ' style="opacity:.4"' : ""}><td>${b.id}</td><td>${escapeHtml(b.shelf)} · ${b.position}</td><td>${reading}</td><td>${dims}</td></tr>`;
    })
    .join("");
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function handle(event) {
  switch (event.type) {
    case "ready": status(`Sweep ${event.sweep_id} · ${event.mock ? "mock flow (offline)" : "live"}`); break;
    case "audio": playPcm(event.data, event.mime); break;
    case "interrupted": stopPlayback(); break;
    case "transcript":
      log(event.speaker, event.text);
      if (mockMode && event.speaker === "agent") speak(event.text);
      break;
    case "turn_complete": closeTurns(); break;
    case "feedback":
      keyframeInFlight = false;
      $("warnings").textContent = event.problems?.length ? `Capture: ${event.problems.join(", ")}` : event.metric ? "" : event.plane ? "No size reference on this unit yet: show the marker" : "";
      break;
    case "inventory": renderBooks(event.books); break;
    case "items":
      $("n-items").textContent = event.item_count;
      event.added.forEach((i) => {
        const row = document.createElement("tr");
        row.innerHTML = `<td>${escapeHtml(i.category)}</td><td>${escapeHtml(i.description)}</td><td>${escapeHtml(i.brand_model || "")}</td><td>${i.dims_cm ? i.dims_cm.join(" × ") + " cm" : ""}</td>`;
        $("items").appendChild(row);
      });
      break;
    case "ar_point": if (event.applied) $("n-points").textContent = Number($("n-points").textContent) + 1; break;
    case "locale": status(`Country ${event.country} · ${event.currency}`); break;
    case "phase": if (event.phase === "processing") { status("Building claim packet…"); stopCapture(); } break;
    case "packet":
      status(`Packet ready: ${event.totals.book_count} books, ${event.review_count} to review`);
      $("packet").href = `/sweeps/${event.sweep_id}/report.html`;
      $("packet").style.display = "inline";
      break;
    case "stage_error": console.warn("stage error", event); break;
    case "error": status(event.message); break;
  }
}

function stopCapture() {
  timers.forEach(clearInterval);
  timers = [];
  $("end").disabled = true;
}

// ---------- demo camera ----------
// Plays the built-in synthetic sweep as the camera, so the whole live flow can be tried without a bookshelf.
// Each picture is held like a slow pan; keyframes are sent once per picture, as a moving camera would give.
const DEMO_HOLD_MS = 2500;
let demoFrame = -1;

async function demoPlayer(frames) {
  const images = await Promise.all(frames.map((src) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`demo frame ${src} did not load`));
    img.src = src;
  })));
  const canvas = document.createElement("canvas");
  canvas.width = images[0].naturalWidth;
  canvas.height = images[0].naturalHeight;
  const ctx = canvas.getContext("2d");
  const show = (i) => { demoFrame = i; ctx.drawImage(images[i], 0, 0); };
  show(0);
  // Frames for the server come from the pictures themselves: the preview <video> shows black until
  // the captured stream delivers, and the first dozen keyframes of a demo were all black.
  grabFrame = (maxSide, quality) => grab(maxSide > 1000 ? $("large") : $("small"), maxSide, quality, canvas);
  return { stream: canvas.captureStream(10), show, count: images.length };
}

async function demoStream(frames) {
  const player = await demoPlayer(frames);
  let i = 0;
  timers.push(setInterval(() => { if (i < player.count - 1) player.show(++i); }, DEMO_HOLD_MS));
  return player.stream;
}

// ---------- mock flow ----------
// The whole journey with no keys and no network: the server replays recorded model answers and a
// scripted agent; this page plays the demo footage, answers the agent's questions, marks the room's
// corners and ends the sweep, speaking the agent's lines with the browser's own voice.
const MOCK_ROOM = [[0, 0, 0], [4.2, 0, 0], [4.2, 0, 3.6], [0, 0, 3.6]]; // a 4.2 x 3.6 m room
const MOCK_CEILING_M = 2.7;
let mockMode = false;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

async function startMock() {
  ["start", "start-ar", "start-demo", "start-mock"].forEach((id) => { $(id).disabled = true; });
  status("Starting mock flow…");
  mockMode = true;
  const manifest = await fetch("/api/demo").then((r) => r.json());
  const player = await demoPlayer(manifest.frames);
  $("preview").srcObject = player.stream;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${scheme}://${location.host}/ws/sweep?mock=1`);
  ws.onmessage = (e) => { const event = JSON.parse(e.data); handle(event); mockStep(event, player); };
  ws.onclose = () => { status("Disconnected"); stopCapture(); };
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  $("end").disabled = false;
}

async function mockStep(event, player) {
  if (event.type === "ready") {
    await wait(4000);
    send({ type: "text", text: "I'm in India." });
  } else if (event.type === "locale") {
    await wait(3000);
    for (let i = 0; i < player.count && ws.readyState === WebSocket.OPEN; i++) {
      player.show(i);
      send({ type: "keyframe", demo_index: i });
      await wait(DEMO_HOLD_MS);
    }
    for (const corner of MOCK_ROOM) {
      send({ type: "ar_point", kind: "floor_corner", position: corner });
      await wait(900);
    }
    send({ type: "ar_point", kind: "ceiling", position: [1, MOCK_CEILING_M, 1] });
    await wait(1500);
    send({ type: "end" });
  } else if (event.type === "transcript" && event.speaker === "agent" && /original or a print/i.test(event.text)) {
    await wait(2500);
    send({ type: "text", text: "It's a print." });
  }
}

async function start({ ar = false, demo = false } = {}) {
  $("start").disabled = true;
  $("start-ar").disabled = true;
  $("start-demo").disabled = true;
  status("Starting camera and microphone…");
  const audio = { echoCancellation: true, noiseSuppression: true, channelCount: 1 };
  if (demo) {
    const manifest = await fetch("/api/demo").then((r) => r.json());
    const mic = await navigator.mediaDevices.getUserMedia({ audio });
    const video = await demoStream(manifest.frames);
    mediaStream = new MediaStream([...mic.getAudioTracks(), ...video.getVideoTracks()]);
    $("preview").srcObject = mediaStream;
  } else if (ar) {
    // WebXR owns the camera in AR; only the microphone comes from getUserMedia.
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio });
    const { startAr } = await import("/static/ar.js");
    grabFrame = await startAr({ onPoint: (kind, position) => send({ type: "ar_point", kind, position }), onEnd: () => send({ type: "end" }) });
    $("ar-controls").style.display = "inline-flex";
  } else {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio,
      // The camera picked in the list (a laptop can have several, e.g. OBS Virtual Camera); else the back camera.
      video: $("camera").value
        ? { deviceId: { exact: $("camera").value }, width: { ideal: 1920 }, height: { ideal: 1080 } }
        : { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
    $("preview").srcObject = mediaStream;
    listCameras();
  }
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const marker = $("marker-cm").value ? `marker_cm=${encodeURIComponent($("marker-cm").value)}` : "";
  const query = [demo ? "demo=1" : "", demo ? "" : marker].filter(Boolean).join("&");
  ws = new WebSocket(`${scheme}://${location.host}/ws/sweep${query ? "?" + query : ""}`);
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => { status("Disconnected"); stopCapture(); };
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  await startMic(mediaStream);
  timers.push(setInterval(sendLiveFrame, LIVE_FRAME_MS));
  timers.push(setInterval(maybeSendKeyframe, 100));
  $("end").disabled = false;
}

const fail = (e) => {
  status(`Could not start: ${e.message}`);
  ["start", "start-ar", "start-demo", "start-mock"].forEach((id) => { $(id).disabled = false; });
};
$("start").onclick = () => start().catch(fail);
$("start-ar").onclick = () => start({ ar: true }).catch(fail);
$("start-demo").onclick = () => start({ demo: true }).catch(fail);

// Cameras are named only after the page has camera permission; the list refills once a sweep starts.
async function listCameras() {
  const select = $("camera");
  const chosen = select.value;
  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  select.replaceChildren(new Option("Default camera", ""),
    ...cameras.map((c, i) => new Option(c.label || `Camera ${i + 1}`, c.deviceId)));
  select.value = chosen;
  return cameras;
}

// Opening the list the first time asks for camera permission, so the cameras get their names
// (before it, a browser reports a single unnamed camera).
$("camera").addEventListener("pointerdown", async () => {
  const cameras = await listCameras();
  if (cameras.some((c) => c.label)) return;
  try {
    const probe = await navigator.mediaDevices.getUserMedia({ video: true });
    probe.getTracks().forEach((t) => t.stop());
    await listCameras();
  } catch { /* permission refused: the default camera is used */ }
}, { once: true });
navigator.mediaDevices?.addEventListener?.("devicechange", listCameras);
listCameras().catch(() => {});
fetch("/api/health").then((r) => r.json()).then((h) => { $("marker-cm").value = h.marker_size_cm; }).catch(() => {});
$("start-mock").onclick = () => startMock().catch(fail);
fetch("/api/demo").then((r) => r.json()).then((m) => {
  if (m.frames?.length) $("start-demo").style.display = "inline-block";
  if (m.mock) $("start-mock").style.display = "inline-block";
}).catch(() => {});
navigator.xr?.isSessionSupported("immersive-ar").then((ok) => { if (ok) $("start-ar").style.display = "inline-block"; });
$("end").onclick = () => send({ type: "end" });
window.claimSweep = { send, handle }; // used by the AR module and tests
