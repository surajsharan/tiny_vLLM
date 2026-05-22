/* tiny_vllm — demo page client.
 *
 * Two streams in play:
 *
 *   /engine/events    — engine state snapshots (one per scheduling step)
 *   /generate         — token-level deltas for whatever prompt this page sent
 *
 * The page itself is stateless; everything is driven by what comes off the
 * event stream.  Token deltas from /generate are merged into per-request UI.
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
};

function logLine(html, cls = "") {
  const t = new Date().toLocaleTimeString();
  ui.log.innerHTML += `<span class="${cls}">[${t}] ${html}</span>\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
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
    if (rc === 0) {
      cls += hashed ? " cached" : " free";
    } else if (rc === 1) {
      cls += " used";
    } else {
      cls += " shared";
    }
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
  // index for later token-delta merges
  state.seqsBySeqId = new Map(all.map(s => [s.seq_id, s]));
  ui.seqs.innerHTML = "";
  if (all.length === 0) {
    ui.seqs.innerHTML = `<div class="muted">(no active sequences — send a prompt above)</div>`;
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
      <div class="seq-text"><span class="prompt">${escapeHtml(promptText)}</span><span class="gen">${escapeHtml(gen)}</span>${s.status === 'running' || s.status === 'prefilling' ? '<span class="cursor">&nbsp;</span>' : ''}</div>
    `;
    ui.seqs.appendChild(div);
  }
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}

function handleEvent(ev) {
  if (ev.type === "snapshot") {
    const snap = ev.payload;
    ui.model.textContent = `· ${snap.config.model}`;
    renderPool(snap.block_pool);
    renderSeqs(snap);
    return;
  }
  if (ev.type === "step") {
    const p = ev.payload;
    ui.statTokens.textContent = p.num_tokens;
    ui.statPfDec.textContent = `${p.num_prefill_seqs} / ${p.num_decode_seqs}`;
    ui.statMs.textContent = p.duration_ms.toFixed(1);
    if (p.preempted?.length) state.preempted += p.preempted.length;
    ui.statPre.textContent = state.preempted;
    renderPool(p.snapshot.block_pool);
    renderSeqs(p.snapshot);

    let msg = `step ${ev.step}: ${p.num_tokens}t (${p.num_prefill_seqs}P/${p.num_decode_seqs}D) in ${p.duration_ms.toFixed(1)}ms`;
    let cls = "ev-step";
    if (p.newly_admitted?.length) {
      msg += ` · admitted seq=${p.newly_admitted.join(",")}`;
      cls = "ev-admit";
    }
    if (p.finished?.length) {
      msg += ` · finished ${p.finished.map(r => r.slice(0,8)).join(",")}`;
      cls = "ev-finish";
    }
    if (p.preempted?.length) {
      msg += ` · PREEMPTED seq=${p.preempted.join(",")}`;
      cls = "ev-preempt";
    }
    logLine(msg, cls);
  }
}

function connectEvents() {
  const es = new EventSource("/engine/events");
  es.onopen = () => {
    ui.connection.textContent = "connected";
    ui.connection.classList.remove("offline");
    ui.connection.classList.add("online");
  };
  es.onerror = () => {
    ui.connection.textContent = "disconnected";
    ui.connection.classList.remove("online");
    ui.connection.classList.add("offline");
  };
  es.onmessage = (e) => {
    if (!e.data) return;
    try {
      handleEvent(JSON.parse(e.data));
    } catch (err) {
      console.error("bad event", err, e.data);
    }
  };
}

async function sendPrompt(prompt) {
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

  // Parse SSE manually so we can read each event as it arrives.
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
        // Repaint the matching seq card if visible.
        const card = document.getElementById(`seq-${myReqId}`);
        if (card) {
          const text = card.querySelector(".seq-text .gen");
          if (text) text.textContent = rec.generated;
        }
      } catch (e) {
        console.error("bad chunk", e, data);
      }
    }
  }
}

ui.send.addEventListener("click", () => sendPrompt($("prompt").value));
ui.sendTwice.addEventListener("click", async () => {
  const p = $("prompt").value;
  // First send fills the prefix cache; second send should hit it.
  await sendPrompt(p);
  await new Promise(r => setTimeout(r, 200));
  await sendPrompt(p);
});
$("prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    sendPrompt(e.target.value);
  }
});

connectEvents();
