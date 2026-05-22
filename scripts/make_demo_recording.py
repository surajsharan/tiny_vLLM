"""Fabricate a representative events.jsonl for the GH Pages demo.

We don't need a model or torch here — we just synthesize a sequence of engine
events that exercises:

  * a prompt being admitted (request event + first step admitting it)
  * chunked prefill across two steps
  * decoding several tokens
  * a second request that hits the prefix cache on identical prefix tokens
  * a finished sequence and a freed-but-cached block

The output is read by `web/app.js`'s replay mode.

Run:
    python scripts/make_demo_recording.py > web/events.jsonl
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy

NUM_BLOCKS = 24
BLOCK_SIZE = 8
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

CONFIG = {
    "model": MODEL,
    "block_size": BLOCK_SIZE,
    "num_blocks": NUM_BLOCKS,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 32,
    "prefix_caching": True,
}

# State we evolve.
ref = [0] * NUM_BLOCKS
hashed = [False] * NUM_BLOCKS
lookups = 0
hits = 0

events: list[dict] = []
ts = 0.0
step = 0


def pool_snapshot() -> dict:
    return {
        "num_blocks": NUM_BLOCKS,
        "block_size": BLOCK_SIZE,
        "num_free_blocks": sum(1 for r in ref if r == 0),
        "num_cached_entries": sum(1 for h in hashed if h),
        "prefix_cache_hits": hits,
        "prefix_cache_lookups": lookups,
        "ref_counts": list(ref),
        "hashed": list(hashed),
    }


def seq_view(rid, sid, status, prompt_len, num_gen, num_computed, num_cached, block_table) -> dict:
    return {
        "seq_id": sid, "request_id": rid, "status": status,
        "prompt_len": prompt_len, "num_generated": num_gen,
        "num_computed_tokens": num_computed,
        "num_cached_prefix_tokens": num_cached,
        "block_table": list(block_table),
    }


def emit(ev_type: str, payload: dict) -> None:
    global ts
    events.append({"type": ev_type, "step": step, "timestamp": round(ts, 3), "payload": payload})


def emit_snapshot(running, waiting) -> None:
    emit("snapshot", {
        "step": step,
        "config": CONFIG,
        "block_pool": pool_snapshot(),
        "running": running,
        "waiting": waiting,
    })


def emit_step(dur_ms, n_tok, n_pf, n_dec, deltas, running, waiting,
              admitted=None, finished=None, preempted=None):
    emit("step", {
        "duration_ms": dur_ms,
        "num_tokens": n_tok,
        "num_seqs": n_pf + n_dec,
        "num_prefill_seqs": n_pf,
        "num_decode_seqs": n_dec,
        "deltas": deltas or [],
        "newly_admitted": admitted or [],
        "finished": finished or [],
        "preempted": preempted or [],
        "snapshot": {
            "step": step,
            "config": CONFIG,
            "block_pool": pool_snapshot(),
            "running": running,
            "waiting": waiting,
        },
    })


# ---------- script the session ----------

# Initial empty snapshot.
emit_snapshot(running=[], waiting=[])
ts += 0.4

# --- request A admitted with a 20-token prompt --------------------------
RID_A = "demo-aaaa1111"
SID_A = 1
PROMPT_A = "Explain paged attention in two sentences. Then explain prefix caching."

step = 1
ts += 0.1
emit("request", {"request_id": RID_A, "seq_id": SID_A, "prompt": PROMPT_A,
                 "prompt_len": 20, "max_tokens": 24})

# Step 1: chunked prefill, first chunk (16 tokens → 2 blocks).
ref[0] = ref[1] = 1
hashed[0] = hashed[1] = True
ts += 0.05
running_A_pf1 = [seq_view(RID_A, SID_A, "prefilling", 20, 0, 16, 0, [0, 1])]
emit_step(280, 16, 1, 0, [], [], [running_A_pf1[0]], admitted=[SID_A])
# (waiting moved to running at end of prefill; here still prefilling, so it
# stays in waiting in our scheduler; emit fix:)
events[-1]["payload"]["snapshot"]["running"] = []
events[-1]["payload"]["snapshot"]["waiting"] = running_A_pf1
ts += 0.3

# Step 2: finish prefill (4 more tokens). Need 1 more block.
step = 2
ref[2] = 1  # block 2 holds last 4 tokens (partial)
ts += 0.05
running_A = [seq_view(RID_A, SID_A, "running", 20, 0, 20, 0, [0, 1, 2])]
emit_step(180, 4, 1, 0, [], running_A, [])
ts += 0.2

# Steps 3-7: A decodes 5 tokens. Each step appends one slot; block 2 fills
# at token 24, then block 3 starts.
TOKS_A = [" Paged", " attention", " splits", " keys", " and"]
for i, tok in enumerate(TOKS_A):
    step += 1
    ts += 0.12
    n_gen = i + 1
    n_comp = 20 + i  # after fwd: num_computed_tokens increments by 1 each decode
    # Block table grows when crossing boundary.
    bt = [0, 1, 2]
    if 20 + n_gen > 24:
        bt = [0, 1, 2, 3]
        ref[3] = 1
    # Maybe hash block 2 when it just filled (after sampling tok at pos 23 → 24 tokens computed).
    if 20 + n_gen >= 24 and not hashed[2]:
        hashed[2] = True
    running_A = [seq_view(RID_A, SID_A, "running", 20, n_gen, 20 + n_gen,
                          0, bt)]
    emit_step(95, 1, 0, 1,
              [{"request_id": RID_A, "new_text": tok, "finished": False, "finish_reason": None}],
              running_A, [])

# --- request B admitted, identical prefix → cache hit -------------------
step += 1
RID_B = "demo-bbbb2222"
SID_B = 2
PROMPT_B = PROMPT_A   # same → prefix cache hit
ts += 0.6
emit("request", {"request_id": RID_B, "seq_id": SID_B, "prompt": PROMPT_B,
                 "prompt_len": 20, "max_tokens": 24})

step += 1
ts += 0.05
# Hit lookups: 2 full blocks of B's prompt (block_size=8 → 16/8=2 blocks).
# Both hit because A's first 2 blocks are hashed (and currently in use).
lookups += 2
hits += 2
ref[0] += 1
ref[1] += 1
# B needs a 3rd block (partial, 4 tokens). Fresh: block 4.
ref[4] = 1
running_B_pf = [seq_view(RID_B, SID_B, "prefilling", 20, 0, 16, 16, [0, 1, 4])]
# A is still running and decoding in this step.
step_A_n_gen = len(TOKS_A)
running_A_now = [seq_view(RID_A, SID_A, "running", 20, step_A_n_gen,
                          20 + step_A_n_gen, 0, [0, 1, 2, 3])]
emit_step(140, 5, 1, 1,
          [{"request_id": RID_A, "new_text": " values", "finished": False, "finish_reason": None}],
          [running_A_now[0]], [running_B_pf[0]], admitted=[SID_B])
ts += 0.12

# Step: B finishes prefill (4 tokens), A decodes one more.
step += 1
running_A_now = [seq_view(RID_A, SID_A, "running", 20, step_A_n_gen + 1,
                          21 + step_A_n_gen, 0, [0, 1, 2, 3])]
running_B = [seq_view(RID_B, SID_B, "running", 20, 0, 20, 16, [0, 1, 4])]
emit_step(110, 5, 1, 1,
          [{"request_id": RID_A, "new_text": " into", "finished": False, "finish_reason": None}],
          [running_A_now[0], running_B[0]], [])
ts += 0.12

# Steps: both decode in lock-step for a few rounds.
A_tokens = [" small", ",", " fixed", "-size", " blocks", " stored", " in"]
B_tokens = [" Prefix", " caching", " reuses", " those", " blocks", " across", " requests"]
for i in range(7):
    step += 1
    ts += 0.10
    a_gen = step_A_n_gen + 2 + i
    b_gen = i + 1
    a_bt = [0, 1, 2, 3]
    b_bt = [0, 1, 4]
    # B crosses block boundary at b_gen such that 20 + b_gen > 24 → needs block 5.
    if 20 + b_gen > 24 and ref[5] == 0:
        ref[5] = 1
    if 20 + b_gen > 24:
        b_bt = [0, 1, 4, 5]
    if 20 + b_gen >= 24 and not hashed[4]:
        hashed[4] = True
    seq_a = seq_view(RID_A, SID_A, "running", 20, a_gen, 20 + a_gen, 0, a_bt)
    seq_b = seq_view(RID_B, SID_B, "running", 20, b_gen, 20 + b_gen, 16, b_bt)
    emit_step(105, 2, 0, 2,
              [{"request_id": RID_A, "new_text": A_tokens[i], "finished": False, "finish_reason": None},
               {"request_id": RID_B, "new_text": B_tokens[i], "finished": False, "finish_reason": None}],
              [seq_a, seq_b], [])

# A finishes (hits EOS or max_tokens).
step += 1
ts += 0.10
ref[0] -= 1   # A releases its blocks; block 0,1 still held by B so refcount drops to 1
ref[1] -= 1
ref[2] = 0    # block 2 (A only) → goes to cached free list (hashed)
ref[3] = 0    # block 3 (A only) → if hashed, cached; else uncached free
# Block 2 was hashed earlier; block 3 we never hashed (partial in A's decode). For demo,
# hash block 3 to make the cache view richer.
hashed[3] = True
seq_b = seq_view(RID_B, SID_B, "running", 20, 8, 28, 16, [0, 1, 4, 5])
emit_step(85, 1, 0, 1,
          [{"request_id": RID_A, "new_text": " the GPU.", "finished": True, "finish_reason": "stop"}],
          [seq_b], [], finished=[RID_A])

# B finishes too.
step += 1
ts += 0.10
ref[0] -= 1
ref[1] -= 1
ref[4] = 0
ref[5] = 0
hashed[5] = True
emit_step(80, 1, 0, 1,
          [{"request_id": RID_B, "new_text": " for the same prompts.", "finished": True, "finish_reason": "stop"}],
          [], [], finished=[RID_B])


# ---------- emit ----------

out = sys.stdout
for ev in events:
    out.write(json.dumps(ev, separators=(",", ":")))
    out.write("\n")
