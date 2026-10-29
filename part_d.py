"""
part_d.py -- Instamart Capstone Part D
========================================
Policy RAG + ReAct Refund Agent

Pipeline:
  1. Policy ingestion: read 5 txt files -> chunk (40 words / 10-word overlap)
                      -> TfidfVectorizer -> retrieve(query, k=3)
  2. TOOL_MAP: get_order, search_policy, calculate_refund,
               issue_refund, escalate_to_human
  3. ReAct agent loop: max_iterations=6, token_limit=6_000,
                       time_limit_seconds=10, safe fallbacks
  4. Batch run over agent_scenarios.csv
     -> outputs/agent_runs.jsonl
     -> outputs/refund_ledger.jsonl  (appended)
  5. Checkpoint / resume via state/ folder

Allowed libraries: standard Python, pandas, numpy, scikit-learn, mock_llm (local)
"""

from __future__ import annotations

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm
from mock_llm import chat, est_tokens

# ─────────────────────────────────────────────────────────────
# 0.  Config
# ─────────────────────────────────────────────────────────────
RAW        = os.path.join("data", "raw")
POLICIES   = os.path.join("data", "policies")
OUTPUTS    = "outputs"
STATE_DIR  = "state"

os.makedirs(OUTPUTS,   exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

CHUNK_SIZE       = 40    # words per chunk
CHUNK_OVERLAP    = 10    # overlap words between consecutive chunks
TOP_K            = 3     # default retrieve k

MAX_ITERATIONS   = 6
TOKEN_LIMIT      = 6_000
TIME_LIMIT_SEC   = 10
LATE_MINUTES     = 45    # delivery > 45 min -> 100% refund
REFUND_THRESHOLD = 1_000 # INR -- above this -> escalate to human

AGENT_RUNS_PATH  = os.path.join(OUTPUTS, "agent_runs.jsonl")
LEDGER_PATH      = os.path.join(OUTPUTS, "refund_ledger.jsonl")
SCENARIOS_PATH   = os.path.join(RAW,     "agent_scenarios.csv")

SEP  = "=" * 65
SEP2 = "-" * 65

# ─────────────────────────────────────────────────────────────
# 1.  Policy RAG
# ─────────────────────────────────────────────────────────────
print(SEP)
print("STEP 1 - Policy ingestion and RAG index build")
print(SEP)


def _chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into word-based windows.
    Each chunk has `size` words; consecutive chunks share `overlap` words.
    """
    words = text.split()
    if not words:
        return []
    step   = size - overlap
    chunks = []
    start  = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def load_policies(policy_dir):
    """
    Read all .txt files in policy_dir.
    Returns list of {"source": filename, "chunk_id": int, "text": str}.
    """
    records = []
    for fname in sorted(os.listdir(policy_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(policy_dir, fname)
        with open(fpath, encoding="utf-8") as fh:
            raw = fh.read().strip()
        for i, chunk in enumerate(_chunk_text(raw)):
            records.append({"source": fname, "chunk_id": i, "text": chunk})
    return records


# Load and index
policy_records = load_policies(POLICIES)
policy_texts   = [r["text"] for r in policy_records]

tfidf_vec    = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tfidf_matrix = tfidf_vec.fit_transform(policy_texts)   # (n_chunks, n_features)

print(f"  Policy files loaded  : {len(set(r['source'] for r in policy_records))}")
print(f"  Total chunks         : {len(policy_records)}")
print(f"  TF-IDF vocab size    : {len(tfidf_vec.vocabulary_):,}")
print()
for src in sorted(set(r["source"] for r in policy_records)):
    n = sum(1 for r in policy_records if r["source"] == src)
    print(f"  {src:<30s}  {n} chunk(s)")


def retrieve(query, k=TOP_K):
    """
    Return top-k policy chunks most relevant to query.
    Each result dict: source, chunk_id, text, score.
    """
    if not query or not query.strip():
        return []
    q_vec   = tfidf_vec.transform([query])
    scores  = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_idx:
        rec = dict(policy_records[idx])
        rec["score"] = float(scores[idx])
        results.append(rec)
    return results


# Quick self-test
_demo = retrieve("refund late delivery", k=2)
print()
print("  RAG self-test (query='refund late delivery', k=2):")
for r in _demo:
    print(f"    [{r['source']}  chunk {r['chunk_id']}]  score={r['score']:.4f}")
    print(f"      {r['text'][:80]}...")

# ─────────────────────────────────────────────────────────────
# 2.  Database helper -- load orders
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 2 - Load orders table")
print(SEP)

orders_df = pd.read_csv(os.path.join(RAW, "orders.csv"))
orders_df["order_amount"]     = pd.to_numeric(orders_df["order_amount"],     errors="coerce")
orders_df["delivery_minutes"] = pd.to_numeric(orders_df["delivery_minutes"], errors="coerce")

# O(1) lookup dict
_orders_map = {row["order_id"]: row.to_dict() for _, row in orders_df.iterrows()}
print(f"  Orders loaded: {len(_orders_map):,}")


# ─────────────────────────────────────────────────────────────
# 3.  TOOL_MAP
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 3 - TOOL_MAP")
print(SEP)


def _tool_get_order(order_id):
    """Fetch order details from in-memory lookup."""
    row = _orders_map.get(order_id)
    if row is None:
        return {"error": f"Order {order_id!r} not found."}
    return {
        "order_id":         row["order_id"],
        "customer_id":      row["customer_id"],
        "status":           row["status"],
        "order_amount":     float(row["order_amount"]) if pd.notna(row["order_amount"]) else 0.0,
        "delivery_minutes": float(row["delivery_minutes"]) if pd.notna(row["delivery_minutes"]) else 0.0,
    }


def _tool_search_policy(query):
    """Run policy RAG and return top-k chunks concatenated."""
    chunks = retrieve(query, k=TOP_K)
    if not chunks:
        return {"result": "No relevant policy found.", "chunks": []}
    context = " | ".join(c["text"] for c in chunks)
    return {
        "result": context,
        "chunks": [{"source": c["source"], "chunk_id": c["chunk_id"],
                    "score": round(c["score"], 4)} for c in chunks],
    }


def _tool_calculate_refund(order_amount, delivery_minutes):
    """
    Refund rules:
      delivery_minutes > 45  => 100% refund
      delivery_minutes <= 45 => 0% refund
    """
    order_amount     = float(order_amount)
    delivery_minutes = float(delivery_minutes)
    if delivery_minutes > LATE_MINUTES:
        refund_inr = round(order_amount, 2)
        reason = (f"Delivery took {delivery_minutes:.1f} min "
                  f"(>{LATE_MINUTES} min threshold) -> 100% refund")
    else:
        refund_inr = 0.0
        reason = (f"Delivery took {delivery_minutes:.1f} min "
                  f"(<={LATE_MINUTES} min threshold) -> no refund")
    return {"refund_inr": refund_inr, "reason": reason}


def _tool_issue_refund(order_id, refund_inr, _ledger_sink=None):
    """Issue a refund for orders <= INR 1000. Appends to ledger sink."""
    entry = {
        "order_id":   order_id,
        "refund_inr": round(float(refund_inr), 2),
        "action":     "ISSUED",
        "ts":         time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if _ledger_sink is not None:
        _ledger_sink.append(entry)
    return {"status": "success", "refund_inr": round(float(refund_inr), 2)}


def _tool_escalate_to_human(order_id, refund_inr, _ledger_sink=None):
    """Escalate refund > INR 1000 to a human support lead."""
    entry = {
        "order_id":   order_id,
        "refund_inr": round(float(refund_inr), 2),
        "action":     "ESCALATED",
        "ts":         time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if _ledger_sink is not None:
        _ledger_sink.append(entry)
    return {
        "status":     "escalated",
        "refund_inr": round(float(refund_inr), 2),
        "message":    "Sent to human support lead for approval.",
    }


# Tool schema definitions passed to mock_llm
TOOL_DEFINITIONS = [
    {
        "name":        "get_order",
        "description": "Fetch order details (status, amount, delivery_minutes) by order_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string",
                             "description": "Order identifier, e.g. ORD-100048"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name":        "search_policy",
        "description": "Search the policy knowledge base for relevant chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Natural language query about policy"}
            },
            "required": ["query"],
        },
    },
    {
        "name":        "calculate_refund",
        "description": "Calculate refund amount given order_amount and delivery_minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount":     {"type": "number"},
                "delivery_minutes": {"type": "number"},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name":        "issue_refund",
        "description": "Issue an approved refund for a completed order (<= INR 1000).",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id":   {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name":        "escalate_to_human",
        "description": "Escalate a refund > INR 1000 to the human support lead.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id":   {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

TOOL_MAP = {
    "get_order":         _tool_get_order,
    "search_policy":     _tool_search_policy,
    "calculate_refund":  _tool_calculate_refund,
    "issue_refund":      _tool_issue_refund,
    "escalate_to_human": _tool_escalate_to_human,
}

print(f"  Tools registered: {list(TOOL_MAP.keys())}")


# ─────────────────────────────────────────────────────────────
# 4.  Checkpoint / Resume helpers
# ─────────────────────────────────────────────────────────────

def _checkpoint_path(scenario_id):
    return os.path.join(STATE_DIR, f"{scenario_id}.json")


def _save_checkpoint(scenario_id, state):
    with open(_checkpoint_path(scenario_id), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, default=str)


def _load_checkpoint(scenario_id):
    path = _checkpoint_path(scenario_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _checkpoint_done(scenario_id):
    """Return True if this scenario has a COMPLETED checkpoint."""
    state = _load_checkpoint(scenario_id)
    return state is not None and state.get("status") == "COMPLETED"


# ─────────────────────────────────────────────────────────────
# 5.  ReAct Agent Loop
# ─────────────────────────────────────────────────────────────

SYSTEM_AGENT = (
    "You are an AGENT for Instamart customer support. "
    "You help customers with refund requests by calling tools in sequence. "
    "Always retrieve the order first, then check policy, then calculate "
    "and issue or escalate the refund."
)


def _dispatch_tool(name, args, ledger_sink):
    """Execute a tool and return its JSON result string."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name!r}"})
    if name in ("issue_refund", "escalate_to_human"):
        result = fn(_ledger_sink=ledger_sink, **args)
    else:
        result = fn(**args)
    return json.dumps(result, default=str)


def _classify_terminal(content):
    """Map a terminal LLM text response to a human-readable outcome label."""
    if content is None:
        return "UNKNOWN"
    upper = content.upper()
    if "REFUND:" in upper:
        return "REFUND"
    if "ESCALATE" in upper:
        return "ESCALATE"
    if "DENIED:" in upper:
        return "DENIED"
    if "NOT_FOUND" in upper:
        return "NOT_FOUND"
    if "NEEDS_INFO" in upper:
        return "NEEDS_INFO"
    return "UNKNOWN"


def run_agent(scenario_id, customer_id, message,
              max_iterations=MAX_ITERATIONS,
              token_limit=TOKEN_LIMIT,
              time_limit_seconds=TIME_LIMIT_SEC):
    """
    Run the ReAct agent loop for a single customer scenario.

    Safeguards: max_iterations, token_limit, time_limit_seconds.

    Returns dict: scenario_id, customer_id, outcome, terminal_message,
                  iterations, tokens_used, elapsed_sec, ledger_entries, trace.
    """
    ledger_sink   = []
    trace         = []
    tokens_used   = 0
    start_ts      = time.monotonic()
    messages      = [
        {"role": "system", "content": SYSTEM_AGENT},
        {"role": "user",   "content": message},
    ]

    def _elapsed():
        return time.monotonic() - start_ts

    outcome          = "UNKNOWN"
    terminal_message = None
    iteration        = 0

    for iteration in range(1, max_iterations + 1):

        # -- Time guard ----------------------------------------
        if _elapsed() >= time_limit_seconds:
            outcome          = "TIMEOUT"
            terminal_message = (f"Agent timed out after {_elapsed():.1f}s "
                                f"(limit={time_limit_seconds}s).")
            trace.append({"iteration": iteration, "event": "TIMEOUT",
                          "detail": terminal_message})
            break

        # -- Token guard ---------------------------------------
        ctx_tokens = sum(
            est_tokens(m.get("content") or
                       json.dumps(m.get("tool_calls", ""), default=str))
            for m in messages
        )
        if tokens_used + ctx_tokens > token_limit:
            outcome          = "TOKEN_LIMIT"
            terminal_message = (f"Token limit ({token_limit}) reached "
                                f"after {tokens_used} tokens.")
            trace.append({"iteration": iteration, "event": "TOKEN_LIMIT",
                          "detail": terminal_message})
            break

        # -- LLM call ------------------------------------------
        try:
            resp = chat(messages, tools=TOOL_DEFINITIONS)
        except Exception as exc:
            outcome          = "LLM_ERROR"
            terminal_message = str(exc)
            trace.append({"iteration": iteration, "event": "LLM_ERROR",
                          "detail": str(exc)})
            break

        tokens_used += getattr(resp.usage, "total_tokens", 0)
        choice       = resp.choices[0]
        finish       = choice.finish_reason
        msg_obj      = choice.message

        # -- Terminal text response ----------------------------
        if finish == "stop" or not msg_obj.tool_calls:
            terminal_message = msg_obj.content
            outcome          = _classify_terminal(terminal_message)
            trace.append({
                "iteration": iteration,
                "event":     "STOP",
                "content":   terminal_message,
                "outcome":   outcome,
            })
            break

        # -- Tool calls ----------------------------------------
        # Build serialisable assistant entry
        asst_tool_calls = [
            {
                "id":        tc.id,
                "name":      tc.name,
                "arguments": (tc.arguments if isinstance(tc.arguments, dict)
                              else json.loads(tc.arguments or "{}")),
            }
            for tc in msg_obj.tool_calls
        ]
        messages.append({
            "role":       "assistant",
            "content":    None,
            "tool_calls": asst_tool_calls,
        })

        for tc in msg_obj.tool_calls:
            tool_name = tc.name
            try:
                raw_args = tc.arguments
                args = (raw_args if isinstance(raw_args, dict)
                        else json.loads(raw_args or "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}

            tool_result_str = _dispatch_tool(tool_name, args, ledger_sink)
            tool_result_obj = json.loads(tool_result_str)

            trace.append({
                "iteration": iteration,
                "event":     "TOOL_CALL",
                "tool":      tool_name,
                "args":      args,
                "result":    tool_result_obj,
            })

            messages.append({
                "role":    "tool",
                "name":    tool_name,
                "content": tool_result_str,
            })
            tokens_used += est_tokens(tool_result_str)

    else:
        # for-loop completed without break -> max iterations exhausted
        outcome          = "MAX_ITER"
        terminal_message = (f"Reached max iterations ({max_iterations}) "
                            "without resolution.")
        trace.append({"iteration": max_iterations, "event": "MAX_ITER",
                      "detail": terminal_message})

    return {
        "scenario_id":      scenario_id,
        "customer_id":      customer_id,
        "message":          message,
        "outcome":          outcome,
        "terminal_message": terminal_message,
        "iterations":       iteration,
        "tokens_used":      tokens_used,
        "elapsed_sec":      round(_elapsed(), 3),
        "ledger_entries":   ledger_sink,
        "trace":            trace,
    }


# ─────────────────────────────────────────────────────────────
# 6.  Batch Run over agent_scenarios.csv
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 4 - Batch agent run over agent_scenarios.csv")
print(SEP)

scenarios = pd.read_csv(SCENARIOS_PATH)
print(f"  Scenarios loaded: {len(scenarios)}")
print(scenarios[["scenario_id", "customer_id", "expected_outcome"]].to_string(index=False))
print()

mock_llm.reset()  # fresh mock counters

run_results   = []
skipped_count = 0
new_count     = 0

# Map expected outcome label -> valid actual outcome labels
_OUTCOME_MAP = {
    "REFUND":    ["REFUND"],
    "ESCALATE":  ["ESCALATE"],
    "DENIED":    ["DENIED"],
    "NOT_FOUND": ["NOT_FOUND"],
    "FALLBACK":  ["TIMEOUT", "MAX_ITER", "TOKEN_LIMIT", "LLM_ERROR"],
}

for _, row in scenarios.iterrows():
    sid = row["scenario_id"]

    # -- Checkpoint / Resume -----------------------------------
    if _checkpoint_done(sid):
        prev = _load_checkpoint(sid)
        prev["_resumed"] = True
        print(f"  [{sid}]  RESUMED from checkpoint  outcome={prev.get('outcome')}")
        run_results.append(prev)
        skipped_count += 1
        continue

    # -- Run agent ---------------------------------------------
    result = run_agent(
        scenario_id=sid,
        customer_id=row["customer_id"],
        message=row["message"],
    )
    result["expected_outcome"] = row["expected_outcome"]
    result["expected_refund"]  = float(row["expected_refund_inr"])

    expected = row["expected_outcome"]
    actual   = result["outcome"]
    result["outcome_match"] = actual in _OUTCOME_MAP.get(expected, [expected])

    # -- Save checkpoint ---------------------------------------
    _save_checkpoint(sid, {**result, "status": "COMPLETED"})

    # -- Write to agent_runs.jsonl -----------------------------
    with open(AGENT_RUNS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, default=str) + "\n")

    # -- Append ledger entries ---------------------------------
    for entry in result.get("ledger_entries", []):
        entry["scenario_id"] = sid
        with open(LEDGER_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    run_results.append(result)
    new_count += 1

    # -- Console summary per scenario --------------------------
    mark      = "[OK]" if result["outcome_match"] else "[!!]"
    led_tag   = ""
    if result["ledger_entries"]:
        le      = result["ledger_entries"][0]
        led_tag = f"  ledger={le['action']} INR {le['refund_inr']:.2f}"
    print(
        f"  [{sid}]  {mark}  actual={actual:<12s}  "
        f"expected={expected:<10s}  "
        f"itr={result['iterations']}  tok={result['tokens_used']}"
        f"{led_tag}"
    )

print()
print(f"  Scenarios run (new)     : {new_count}")
print(f"  Scenarios skipped (ckpt): {skipped_count}")
print(f"  agent_runs.jsonl        : {AGENT_RUNS_PATH}")
print(f"  refund_ledger.jsonl     : {LEDGER_PATH}")


# ─────────────────────────────────────────────────────────────
# 7.  Summary Report
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 5 - Summary report")
print(SEP)

total         = len(run_results)
matched       = sum(1 for r in run_results if r.get("outcome_match"))
refund_rows   = [r for r in run_results if r["outcome"] == "REFUND"]
escalate_rows = [r for r in run_results if r["outcome"] == "ESCALATE"]
denied_rows   = [r for r in run_results if r["outcome"] == "DENIED"]
fallback_rows = [r for r in run_results
                 if r["outcome"] in {"TIMEOUT", "MAX_ITER", "TOKEN_LIMIT", "LLM_ERROR"}]
not_found_rows= [r for r in run_results if r["outcome"] == "NOT_FOUND"]

print(f"  Total scenarios      : {total}")
print(f"  Outcome match        : {matched}/{total}  ({matched/total:.0%})")
print()
print(f"  REFUND               : {len(refund_rows)}")
print(f"  ESCALATE             : {len(escalate_rows)}")
print(f"  DENIED               : {len(denied_rows)}")
print(f"  NOT_FOUND            : {len(not_found_rows)}")
print(f"  FALLBACK/TIMEOUT     : {len(fallback_rows)}")

# Ledger summary
if os.path.exists(LEDGER_PATH):
    with open(LEDGER_PATH, encoding="utf-8") as fh:
        ledger_lines = [l for l in fh.readlines() if l.strip()]
    if ledger_lines:
        ledger_df = pd.DataFrame([json.loads(l) for l in ledger_lines])
        total_issued   = ledger_df[ledger_df["action"] == "ISSUED"]["refund_inr"].sum()
        total_escalate = ledger_df[ledger_df["action"] == "ESCALATED"]["refund_inr"].sum()
        print()
        print(f"  Ledger entries       : {len(ledger_df)}")
        print(f"  Total issued (INR)   : {total_issued:.2f}")
        print(f"  Total escalated (INR): {total_escalate:.2f}")
        print()
        print("  Ledger detail:")
        print(ledger_df[["scenario_id", "order_id", "action", "refund_inr"]].to_string(index=False))

# Trace excerpt for first non-resumed scenario
first_new = next((r for r in run_results if not r.get("_resumed")), None)
if first_new:
    print()
    print(f"  Trace excerpt for [{first_new['scenario_id']}]:")
    for step in first_new.get("trace", []):
        evt = step.get("event", "")
        if evt == "TOOL_CALL":
            print(f"    itr={step['iteration']}  TOOL_CALL  "
                  f"{step['tool']}({step['args']})")
            print(f"           -> {str(step.get('result', {}))[:80]}")
        elif evt == "STOP":
            print(f"    itr={step['iteration']}  STOP  "
                  f"outcome={step.get('outcome')}  "
                  f"msg={step.get('content', '')[:60]!r}")
        else:
            print(f"    itr={step['iteration']}  {evt}  "
                  f"{step.get('detail', '')[:60]}")


# ─────────────────────────────────────────────────────────────
# 8.  Validation checkpoints
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 6 - Validation checkpoints")
print(SEP)


def check(name, cond, detail=""):
    mark = "[OK]" if cond else "[!!]"
    msg  = f"  {mark} [{'PASS' if cond else 'FAIL'}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    assert cond, f"Checkpoint failed: {name}  {detail}"


check("Policy chunks created",
      len(policy_records) >= 5,
      f"{len(policy_records)} chunks from 5 files")

check("TF-IDF matrix matches chunk count",
      tfidf_matrix.shape[0] == len(policy_records),
      f"shape={tfidf_matrix.shape}")

check("retrieve() returns <=k results",
      len(retrieve("refund", k=3)) <= 3)

check("retrieve() non-empty for 'refund'",
      len(retrieve("refund", k=1)) == 1)

check("chunk size is 40 words (sample)",
      len(_chunk_text("word " * 80)[0].split()) == CHUNK_SIZE)

check("chunk overlap is 10 words (sample)",
      _chunk_text("word " * 80)[0].split()[-CHUNK_OVERLAP:] ==
      _chunk_text("word " * 80)[1].split()[:CHUNK_OVERLAP])

check("TOOL_MAP has all 5 tools",
      set(TOOL_MAP.keys()) == {
          "get_order", "search_policy", "calculate_refund",
          "issue_refund", "escalate_to_human"
      })

check("get_order returns error for unknown order",
      "error" in _tool_get_order("ORD-999999"))

check("calculate_refund: >45 min -> 100% refund",
      _tool_calculate_refund(500.0, 60.0)["refund_inr"] == 500.0)

check("calculate_refund: =45 min -> 0 refund",
      _tool_calculate_refund(500.0, 45.0)["refund_inr"] == 0.0)

check("calculate_refund: <45 min -> 0 refund",
      _tool_calculate_refund(500.0, 30.0)["refund_inr"] == 0.0)

_ledger_test = []
_tool_issue_refund("ORD-TEST", 99.0, _ledger_sink=_ledger_test)
check("issue_refund writes to ledger sink",
      len(_ledger_test) == 1 and _ledger_test[0]["action"] == "ISSUED")

_ledger_test2 = []
_tool_escalate_to_human("ORD-TEST", 1500.0, _ledger_sink=_ledger_test2)
check("escalate_to_human writes to ledger sink",
      len(_ledger_test2) == 1 and _ledger_test2[0]["action"] == "ESCALATED")

check("agent_runs.jsonl written",
      os.path.exists(AGENT_RUNS_PATH))

check("all scenario checkpoints exist",
      all(_checkpoint_done(row["scenario_id"])
          for _, row in scenarios.iterrows()))

check("outcome match rate >= 85%",
      matched / total >= 0.85,
      f"{matched}/{total} = {matched/total:.0%}")

check("REFUND scenario has ISSUED ledger entry",
      any(e["action"] == "ISSUED"
          for r in run_results for e in r.get("ledger_entries", [])))

check("ESCALATE scenario has ESCALATED ledger entry",
      any(e["action"] == "ESCALATED"
          for r in run_results for e in r.get("ledger_entries", [])))

print()
print(SEP)
print("ALL CHECKPOINTS PASSED  [OK]")
print(SEP)
print()
print("Part D complete.  Files written:")
print(f"  {AGENT_RUNS_PATH}")
print(f"  {LEDGER_PATH}")
print(f"  state/  ({len(scenarios)} checkpoint files)")
