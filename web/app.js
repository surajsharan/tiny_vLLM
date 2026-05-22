/* tiny_vllm — demo page client.
 *
 * Runs in one of two modes:
 *
 *   LIVE     — talks to a tiny_vllm server.  Subscribes to /engine/events
 *              (SSE) and POSTs to /generate to submit prompts.
 *   REPLAY   — no backend.  Fetches a pre-recorded events.jsonl from the
 *              same directory and dispatches each event with original timing.
 *              Used for the GitHub Pages demo.
 *
 * Mode is auto-detected: we try SSE first; if there's no response within a
 * short window we fall back to replay.  Force a mode with ?mode=replay or
 * ?mode=live in the URL.  Point at a different recording with ?session=URL.
 */

const $ = (id) => document.getElementById(id);

const ui = {
  connection: $("connection"),
  model: $("model"),
  pool: $("block-pool"),
  poolSummary: $("pool-summary"),
  schedStep: $("sched-step"),
  statTokens: $("stat-tokens"),
  statPfDec: $("stat-pfdec"),
  statMs: $("stat-ms"),
  statCache: $("stat-cache"),
  statFree: $("stat-free"),
  statPre: $("stat-pre"),
  log: $("log"),
  seqs: $("seqs"),
  send: $("send"),
  sendTwice: $("send-twice"),
  prompt: $("prompt"),
  banner: $("banner"),
  speed: $("speed"),
  playPause: $("play-pause"),
  restart: $("restart"),
};

const state = {
  poolEls: [],
  numBlocks: 0,
  blockSize: 16,
  preempted: 0,
  // request_id -> { promptText, generated, finished, finishReason }
  requests: new Map(),
  // seq_id -> { request_id, blockTable, cachedPrefixBlocks, status, ... }
  seqsBySeqId: new Map(),
  mode: "connecting",     // "live" | "replay" | "connecting"
  replay: null,           // controller object for replay mode
};

function logLine(html, cls = "") {
  const t = new Date().toLocaleTimeString();
  ui.log.innerHTML += `<span class="${cls}">[${t}] ${html}</span>\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

function setBanner(text, cls) {
  if (!ui.banner) return;
  ui.banner.textContent = text;
  ui.banner.className = `banner ${cls || ""}`;
  ui.banner.style.display = text ? "" : "none";
}

function setMode(mode) {
  state.mode = mode;
  if (mode === "live") {
    ui.connection.textContent = "live";
    ui.connection.classList.remove("offline");
    ui.connection.classList.add("online");
    ui.send.disabled = false;
    ui.sendTwice.disabled = false;
    ui.prompt.disabled = false;
    setBanner("", "");
    if (ui.speed) ui.speed.style.display = "none";
    if (ui.playPause) ui.playPause.style.display = "none";
    if (ui.restart) ui.restart.style.display = "none";
  } else if (mode === "replay") {
    ui.connection.textContent = "replay";
    ui.connection.classList.remove("offline");
    ui.connection.classList.add("replay");
    // Keep textarea editable — feels alive — but the Send buttons can't
    // really submit, so they open the run-locally hint instead.
    ui.send.disabled = false;
    ui.sendTwice.disabled = false;
    ui.prompt.disabled = false;
    ui.send.classList.add("replay-locked");
    ui.sendTwice.classList.add("replay-locked");
    ui.send.title = "Replay mode — click to see how to run live locally";
    ui.sendTwice.title = ui.send.title;
    ui.prompt.placeholder = "Recorded demo — typing won't submit. Run the server locally to use your own prompts.";
    setBanner(
      "REPLAY MODE — this is a pre-recorded session. Send buttons show local-run instructions instead.",
      "replay-banner",
    );
    if (ui.speed) ui.speed.style.display = "";
    if (ui.playPause) ui.playPause.style.display = "";
    if (ui.restart) ui.restart.style.display = "";
  } else {
    ui.connection.textContent = "connecting…";
    ui.connection.classList.add("offline");
  }
}

function initPool(numBlocks) {
  if (state.numBlocks === numBlocks && state.poolEls.length === numBlocks) return;
  state.numBlocks = numBlocks;
  ui.pool.innerHTML = "";
  state.poolEls = [];
  for (let i = 0; i < numBlocks; i++) {
    const el = document.createElement("div");
    el.className = "block free";
    el.title = `block ${i}`;
    ui.pool.appendChild(el);
    state.poolEls.push(el);
  }
}

function renderPool(pool) {
  initPool(pool.num_blocks);
  state.blockSize = pool.block_size;
  for (let i = 0; i < pool.num_blocks; i++) {
    const el = state.poolEls[i];
    const rc = pool.ref_counts[i];
    const hashed = pool.hashed[i];
    let cls = "block";
    if (rc === 0) cls += hashed ? " cached" : " free";
    else if (rc === 1) cls += " used";
    else cls += " shared";
    if (hashed) cls += " hashed";
    el.className = cls;
    el.title = `block ${i} — refcount=${rc}${hashed ? " — hashed (cacheable)" : ""}`;
  }
  ui.poolSummary.textContent =
    `${pool.num_blocks - pool.num_free_blocks}/${pool.num_blocks} used · ` +
    `${pool.num_cached_entries} cached entries · ` +
    `prefix-cache ${pool.prefix_cache_hits}/${pool.prefix_cache_lookups}`;
  ui.statFree.textContent = pool.num_free_blocks;
  if (pool.prefix_cache_lookups > 0) {
    const pct = (100 * pool.prefix_cache_hits / pool.prefix_cache_lookups).toFixed(0);
    ui.statCache.textContent = `${pct}%`;
  } else {
    ui.statCache.textContent = "—";
  }
}

function renderSeqs(snapshot) {
  ui.schedStep.textContent = ` — step ${snapshot.step}`;
  const all = [...snapshot.running, ...snapshot.waiting];
  state.seqsBySeqId = new Map(all.map(s => [s.seq_id, s]));
  ui.seqs.innerHTML = "";
  if (all.length === 0) {
    ui.seqs.innerHTML = `<div class="muted">(no active sequences${state.mode === 'replay' ? '' : ' — send a prompt above'})</div>`;
    return;
  }
  for (const s of all) {
    const reqRec = state.requests.get(s.request_id);
    const promptText = reqRec?.promptText ?? "(prompt elided)";
    const gen = reqRec?.generated ?? "";

    const div = document.createElement("div");
    div.className = "seq";
    div.id = `seq-${s.request_id}`;

    const cachedBlocks = Math.floor(s.num_cached_prefix_tokens / state.blockSize);
    const blocksHTML = s.block_table.map((bid, i) => {
      const klass = i < cachedBlocks ? "seq-block cached-hit"
                  : (snapshot.block_pool.ref_counts[bid] > 1 ? "seq-block shared" : "seq-block");
      return `<div class="${klass}" title="block ${bid}${i < cachedBlocks ? ' (prefix-cache hit)' : ''}">${bid}</div>`;
    }).join("");

    div.innerHTML = `
      <div class="seq-header">
        <span class="seq-id">req=${s.request_id.slice(0, 8)} seq=${s.seq_id}</span>
        <span class="seq-status ${s.status}">${s.status}</span>
        <span class="seq-meta">
          prompt=${s.prompt_len} · generated=${s.num_generated} ·
          cached=${s.num_cached_prefix_tokens}/${s.prompt_len} ·
          blocks=${s.block_table.length}
        </span>
      </div>
      <div class="seq-blocks">${blocksHTML || '<span class="muted">(no blocks yet)</span>'}</div>
      <div class="seq-text"><span class="prompt">${escapeHtml(promptText)}</span><span class="gen">${escapeHtml(gen)}</span>${(s.status === 'running' || s.status === 'prefilling') ? '<span class="cursor">&nbsp;</span>' : ''}</div>
    `;
    ui.seqs.appendChild(div);
  }
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}

function applyDeltas(deltas) {
  if (!deltas) return;
  for (const d of deltas) {
    let rec = state.requests.get(d.request_id);
    if (!rec) {
      rec = { promptText: "(prompt unknown)", generated: "", finished: false };
      state.requests.set(d.request_id, rec);
    }
    if (d.new_text) rec.generated += d.new_text;
    if (d.finished) {
      rec.finished = true;
      rec.finishReason = d.finish_reason;
    }
    const card = document.getElementById(`seq-${d.request_id}`);
    if (card) {
      const t = card.querySelector(".seq-text .gen");
      if (t) t.textContent = rec.generated;
    }
  }
}

function handleEvent(ev) {
  if (ev.type === "snapshot") {
    const snap = ev.payload;
    ui.model.textContent = `· ${snap.config.model}`;
    renderPool(snap.block_pool);
    renderSeqs(snap);
    return;
  }
  if (ev.type === "request") {
    // From the recording: capture prompt text + max_tokens for the UI.
    const p = ev.payload;
    state.requests.set(p.request_id, {
      promptText: p.prompt,
      generated: "",
      finished: false,
    });
    logLine(`request ${p.request_id.slice(0,8)} — prompt=${p.prompt_len}t max_tokens=${p.max_tokens}`, "ev-admit");
    return;
  }
  if (ev.type === "step") {
    const p = ev.payload;
    ui.statTokens.textContent = p.num_tokens;
    ui.statPfDec.textContent = `${p.num_prefill_seqs} / ${p.num_decode_seqs}`;
    ui.statMs.textContent = p.duration_ms.toFixed(1);
    if (p.preempted?.length) state.preempted += p.preempted.length;
    ui.statPre.textContent = state.preempted;
    applyDeltas(p.deltas);
    renderPool(p.snapshot.block_pool);
    renderSeqs(p.snapshot);

    let msg = `step ${ev.step}: ${p.num_tokens}t (${p.num_prefill_seqs}P/${p.num_decode_seqs}D) in ${p.duration_ms.toFixed(1)}ms`;
    let cls = "ev-step";
    if (p.newly_admitted?.length) { msg += ` · admitted seq=${p.newly_admitted.join(",")}`; cls = "ev-admit"; }
    if (p.finished?.length) { msg += ` · finished ${p.finished.map(r => r.slice(0,8)).join(",")}`; cls = "ev-finish"; }
    if (p.preempted?.length) { msg += ` · PREEMPTED seq=${p.preempted.join(",")}`; cls = "ev-preempt"; }
    logLine(msg, cls);
  }
}

// ---------- live mode (SSE) ----------

function connectLive() {
  const es = new EventSource("/engine/events");
  let gotOne = false;
  es.onopen = () => { /* wait for first message to confirm live */ };
  es.onerror = () => {
    if (!gotOne) {
      es.close();
      startReplay();  // fall back
    } else {
      ui.connection.textContent = "disconnected";
      ui.connection.classList.remove("online");
      ui.connection.classList.add("offline");
    }
  };
  es.onmessage = (e) => {
    if (!e.data) return;
    if (!gotOne) { gotOne = true; setMode("live"); }
    try { handleEvent(JSON.parse(e.data)); }
    catch (err) { console.error("bad event", err, e.data); }
  };
  // Give the server a couple seconds to respond before falling back.
  setTimeout(() => {
    if (!gotOne) {
      es.close();
      startReplay();
    }
  }, 2000);
}

// ---------- replay mode ----------

async function startReplay() {
  setMode("replay");
  const params = new URLSearchParams(location.search);
  const url = params.get("session") || "events.jsonl";
  let text;
  try {
    const resp = await fetch(url, { cache: "no-cache" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    text = await resp.text();
  } catch (e) {
    setBanner(
      `Could not load recording (${url}). Run the server locally or commit a web/events.jsonl recording.`,
      "replay-banner error",
    );
    ui.connection.textContent = "no recording";
    return;
  }
  const events = text.split("\n").filter(Boolean).map(l => JSON.parse(l));
  if (events.length === 0) {
    setBanner("Recording is empty.", "replay-banner error");
    return;
  }
  state.replay = new Replayer(events);
  state.replay.start();
}

class Replayer {
  constructor(events) {
    this.events = events;
    this.idx = 0;
    this.speed = parseFloat($("speed")?.value || "1");
    this.paused = false;
    this._timeout = null;
  }
  reset() {
    this.stop();
    this.idx = 0;
    state.requests.clear();
    state.preempted = 0;
    ui.log.innerHTML = "";
  }
  setSpeed(s) {
    this.speed = s;
    if (!this.paused) {
      this.stop();
      this._schedule();
    }
  }
  pause() { this.paused = true; this.stop(); }
  resume() { if (!this.paused) return; this.paused = false; this._schedule(); }
  stop() {
    if (this._timeout) clearTimeout(this._timeout);
    this._timeout = null;
  }
  start() {
    this.reset();
    this._schedule(0);
  }
  _schedule(delayOverride) {
    if (this.idx >= this.events.length) {
      logLine("(replay complete — press Restart to replay)", "ev-finish");
      return;
    }
    let delay = 0;
    if (delayOverride !== undefined) {
      delay = delayOverride;
    } else if (this.idx > 0) {
      const gap = this.events[this.idx].timestamp - this.events[this.idx - 1].timestamp;
      delay = Math.max(0, Math.min(gap, 1.0)) * 1000 / this.speed;  // cap at 1s
    }
    this._timeout = setTimeout(() => {
      const ev = this.events[this.idx++];
      try { handleEvent(ev); } catch (e) { console.error(e); }
      if (!this.paused) this._schedule();
    }, delay);
  }
}

// ---------- live: prompt submission ----------

async function sendPrompt(prompt) {
  if (state.mode !== "live") return;
  const body = {
    prompt,
    max_tokens: parseInt($("max_tokens").value, 10),
    temperature: parseFloat($("temperature").value),
    top_p: parseFloat($("top_p").value),
    stream: true,
  };
  const resp = await fetch("/generate", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    logLine(`request failed: ${txt}`, "ev-preempt");
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let myReqId = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      try {
        const j = JSON.parse(data);
        if (!myReqId) {
          myReqId = j.request_id;
          state.requests.set(myReqId, { promptText: prompt, generated: "", finished: false });
        }
        const rec = state.requests.get(myReqId);
        if (j.text) rec.generated += j.text;
        rec.finished = j.finished;
        rec.finishReason = j.finish_reason;
        const card = document.getElementById(`seq-${myReqId}`);
        if (card) {
          const t = card.querySelector(".seq-text .gen");
          if (t) t.textContent = rec.generated;
        }
      } catch (e) { console.error("bad chunk", e, data); }
    }
  }
}

function showRunLocally() {
  const el = $("run-locally");
  if (!el) return;
  el.hidden = false;
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function trySend(handler) {
  if (state.mode !== "live") { showRunLocally(); return; }
  handler();
}

ui.send.addEventListener("click", () => trySend(() => sendPrompt(ui.prompt.value)));
ui.sendTwice.addEventListener("click", () => trySend(async () => {
  const p = ui.prompt.value;
  await sendPrompt(p);
  await new Promise(r => setTimeout(r, 200));
  await sendPrompt(p);
}));
ui.prompt.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") trySend(() => sendPrompt(e.target.value));
});

// Run-locally callout controls.
const rlClose = $("rl-close");
if (rlClose) rlClose.addEventListener("click", () => { $("run-locally").hidden = true; });
const rlCopy = $("rl-copy");
if (rlCopy) rlCopy.addEventListener("click", async () => {
  const cmd = $("run-locally").querySelector("pre").innerText.trim();
  try {
    await navigator.clipboard.writeText(cmd);
    rlCopy.textContent = "Copied ✓";
    setTimeout(() => { rlCopy.textContent = "Copy command"; }, 1500);
  } catch {
    rlCopy.textContent = "Copy failed";
  }
});

if (ui.speed) ui.speed.addEventListener("change", () => {
  state.replay?.setSpeed(parseFloat(ui.speed.value));
});
if (ui.playPause) ui.playPause.addEventListener("click", () => {
  if (!state.replay) return;
  if (state.replay.paused) { state.replay.resume(); ui.playPause.textContent = "Pause"; }
  else { state.replay.pause(); ui.playPause.textContent = "Play"; }
});
if (ui.restart) ui.restart.addEventListener("click", () => state.replay?.start());

// ---------- "Try live" link → Hugging Face Space ----------

(function setupHFLink() {
  const url = document.body.getAttribute("data-hf-space") || "";
  // Don't advertise the live link if we're already on it (avoids
  // showing "try live →" while on the live page).
  const onHF = /\.hf\.space$/i.test(location.hostname) ||
               /huggingface\.co$/i.test(location.hostname);
  if (!url || onHF) return;
  const top = document.getElementById("hf-live");
  if (top) { top.href = url; top.style.display = ""; }
  const rl = document.getElementById("rl-hf");
  if (rl)  { rl.href  = url; rl.style.display  = ""; }
})();

// ---------- entry point ----------

(function boot() {
  setMode("connecting");
  const force = new URLSearchParams(location.search).get("mode");
  if (force === "replay") startReplay();
  else if (force === "live") connectLive();
  else connectLive();   // will auto-fall-back to replay on no-response
})();
