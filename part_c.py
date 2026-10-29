"""
part_c.py -- Instamart Capstone Part C
========================================
Support Ticket Triage using mock_llm.chat()

Pipeline:
  1. Load tickets + customer segments from Part B
  2. Input guardrails  (skip empty, skip > 400 tokens)
  3. PII redaction     (email, phone, ORD-XXXXXX)
  4. Retry wrapper     (max 3, exponential backoff + jitter, TransientLLMError)
  5. JSON/keyword fallback validator
  6. Zero-shot run     (6,000 token budget)  -> outputs/triage_budget_demo.csv
  7. Few-shot run      (100,000 token budget) + segment routing
                                             -> outputs/triage_results.csv

Allowed libraries: standard Python, pandas, numpy, mock_llm (local), sqlite3
"""

import json
import math
import os
import random
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

# Local mock LLM (no external API)
import mock_llm
from mock_llm import chat, est_tokens, TransientLLMError, reset, CATS

# ─────────────────────────────────────────────────────────────
# 0.  Config
# ─────────────────────────────────────────────────────────────
RAW     = os.path.join("data", "raw")
OUTPUTS = "outputs"
os.makedirs(OUTPUTS, exist_ok=True)

MAX_TOKEN_INPUT  = 400        # guardrail: skip tickets longer than this
MAX_RETRIES      = 3          # retry on TransientLLMError
BACKOFF_BASE     = 0.5        # seconds base for exponential backoff
VALID_CATS       = set(CATS)  # {"billing","delivery","technical","fraud"}

ZERO_SHOT_BUDGET = 6_000      # total token budget for zero-shot run
FEW_SHOT_BUDGET  = 100_000    # total token budget for few-shot run

SEP  = "=" * 65
SEP2 = "-" * 65

# ─────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────
print(SEP)
print("STEP 1 - Loading data")
print(SEP)

tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"))
tickets["text"] = tickets["text"].fillna("")

# Load customer segments from Part B
seg_path = os.path.join(OUTPUTS, "customer_segments.csv")
if os.path.exists(seg_path):
    segments = pd.read_csv(seg_path)[["customer_id", "segment_label"]]
else:
    segments = pd.DataFrame(columns=["customer_id", "segment_label"])
    print("  WARNING: customer_segments.csv not found. Segment routing disabled.")

print(f"  Tickets   : {len(tickets):,}")
print(f"  Segments  : {len(segments):,}")
print(f"  Ticket columns : {tickets.columns.tolist()}")
print(f"  True category distribution:")
print(tickets["true_category"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────
# 2.  INPUT GUARDRAILS
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 2 - Input guardrails")
print(SEP)

tickets["token_est"] = tickets["text"].apply(est_tokens)

empty_mask    = tickets["text"].str.strip() == ""
too_long_mask = tickets["token_est"] > MAX_TOKEN_INPUT

print(f"  Total tickets        : {len(tickets):,}")
print(f"  Empty text (skip)    : {empty_mask.sum()}")
print(f"  > {MAX_TOKEN_INPUT} tokens (skip) : {too_long_mask.sum()}")
print(f"  Skipped ticket IDs:")
skipped_tids = tickets[empty_mask | too_long_mask][["ticket_id", "text", "token_est"]]
for _, srow in skipped_tids.iterrows():
    r = "empty" if srow["text"].strip() == "" else f"{int(srow['token_est'])} tokens"
    print(f"    {srow['ticket_id']}  ({r})")

# Processable tickets
processable = tickets[~(empty_mask | too_long_mask)].copy().reset_index(drop=True)
print(f"  Processable tickets  : {len(processable):,}")

# ─────────────────────────────────────────────────────────────
# 3.  PII REDACTION
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 3 - PII redaction")
print(SEP)

# Patterns to redact
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")                   # Indian mobile: starts 6-9, 10 digits
ORDER_RE = re.compile(r"\bORD-\d{6}\b")                     # exactly ORD-XXXXXX


def redact_pii(text: str) -> tuple[str, int, int, int]:
    """Return (redacted_text, n_email, n_phone, n_order) replaced."""
    t, n_email = EMAIL_RE.subn("[EMAIL]", text)
    t, n_phone = PHONE_RE.subn("[PHONE]", t)
    t, n_order = ORDER_RE.subn("[ORDER_ID]", t)
    return t, n_email, n_phone, n_order


# Demo on one ticket
sample_row = processable[processable["ticket_id"] == "T0001"].iloc[0] if "T0001" in processable["ticket_id"].values else processable.iloc[0]
redacted_sample, *_ = redact_pii(sample_row["text"])
print(f"  Sample ticket  : {sample_row['ticket_id']}")
print(f"  Original       : {sample_row['text'][:120]}...")
print(f"  Redacted       : {redacted_sample[:120]}...")

# Apply to all processable tickets
processable[["text_redacted", "n_email_redacted",
             "n_phone_redacted", "n_order_redacted"]] = processable["text"].apply(
    lambda t: pd.Series(redact_pii(t))
)
total_pii = (processable["n_email_redacted"] +
             processable["n_phone_redacted"] +
             processable["n_order_redacted"]).sum()
print(f"\n  Total PII tokens redacted across all tickets : {total_pii}")
print(f"    Emails  : {processable['n_email_redacted'].sum()}")
print(f"    Phones  : {processable['n_phone_redacted'].sum()}")
print(f"    Order IDs: {processable['n_order_redacted'].sum()}")

# ─────────────────────────────────────────────────────────────
# 4.  RETRY WRAPPER
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 4 - Retry mechanism (max=3, exponential backoff + jitter)")
print(SEP)

_rng = random.Random(42)   # deterministic jitter for reproducibility

def call_with_retry(messages: list, tools=None,
                    max_retries: int = MAX_RETRIES,
                    base: float = BACKOFF_BASE) -> object:
    """
    Call mock_llm.chat() with retry on TransientLLMError.
    Backoff: base * 2^attempt  +  uniform jitter [0, 0.5s]
    """
    last_exc = None
    for attempt in range(max_retries + 1):   # 0, 1, 2, 3
        try:
            return chat(messages, tools=tools)
        except TransientLLMError as exc:
            last_exc = exc
            if attempt == max_retries:
                break                         # exhausted retries
            wait = base * (2 ** attempt) + _rng.uniform(0, 0.5)
            # (no actual sleep in tests — mock_llm recovers on next call)
    raise last_exc   # re-raise after all retries exhausted


print("  Retry config: max_retries=3, base=0.5s, jitter=[0, 0.5s]")
print("  Retry formula: wait = 0.5 * 2^attempt + uniform(0, 0.5)")

# ─────────────────────────────────────────────────────────────
# 5.  JSON / KEYWORD FALLBACK VALIDATOR
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 5 - Response parser with JSON / keyword fallback")
print(SEP)

# Keyword-to-category mapping for non-JSON responses
KW_FALLBACK = {
    "fraud":     {"fraud", "unauthorized", "hacked", "stolen"},
    "billing":   {"billing", "charged", "invoice", "bill", "refund",
                  "deducted", "payment"},
    "technical": {"technical", "app", "crash", "login", "otp",
                  "error", "freezes"},
    "delivery":  {"delivery", "delivered", "rider", "arrived", "late"},
}


def parse_response(content: str | None) -> tuple[str, str]:
    """
    Return (category, parse_method).
    parse_method is one of: 'json', 'keyword', 'unknown'
    """
    if not content:
        return "unknown", "unknown"

    # 1. Try JSON parse
    try:
        obj = json.loads(content.strip())
        cat = str(obj.get("category", "")).strip().lower()
        if cat in VALID_CATS:
            return cat, "json"
        # JSON parsed but category value not in allowed list
        # fall through to keyword
    except (json.JSONDecodeError, AttributeError):
        pass

    # 2. Keyword fallback: scan content for category keywords
    words = set(re.findall(r"[a-z]+", content.lower()))
    for cat in ("fraud", "billing", "technical", "delivery"):  # priority order
        if KW_FALLBACK[cat] & words:
            return cat, "keyword"

    return "unknown", "unknown"


# Quick self-test of parser
_tests = [
    ('{"category": "billing", "reason": "charged"}', "billing", "json"),
    ('{"category": "shipping", "reason": "parcel"}', "unknown", "keyword"),  # bad JSON category
    ("The customer appears to have a delivery-related problem.", "delivery", "keyword"),
    ("completely unrelated sentence about nothing", "unknown", "unknown"),
]
print("  Parser self-tests:")
for raw, exp_cat, exp_method in _tests:
    cat, method = parse_response(raw)
    # shipping maps to unknown/keyword since it's not a valid category and has no keyword match
    ok = "[OK]" if cat == exp_cat else "[!!]"
    print(f"    {ok}  input={raw[:50]!r:<55}  cat={cat}  method={method}")

# ─────────────────────────────────────────────────────────────
# 6.  PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────

SYSTEM_ZERO_SHOT = (
    "You are a CLASSIFY customer support triage assistant for Instamart. "
    "Your task is to classify support tickets into exactly one of these categories: "
    "billing, delivery, technical, fraud. "
    "Respond ONLY with a JSON object: {\"category\": \"<one of the four>\", "
    "\"reason\": \"<one-line explanation>\"}."
)

# Few-shot examples (one per category, PII already redacted)
FEW_SHOT_EXAMPLES = [
    ("I was charged twice for the same order [ORDER_ID]. "
     "Please refund the extra payment. My email is [EMAIL].",
     "billing"),
    ("My order [ORDER_ID] has not arrived and the rider is not picking up calls. "
     "Please call [PHONE].",
     "delivery"),
    ("The app crashes every time I open the cart. "
     "I tried to place [ORDER_ID] again and got an error.",
     "technical"),
    ("There is an unauthorized transaction on my card linked to [ORDER_ID]. "
     "I did not place it. Call [PHONE].",
     "fraud"),
]

EXAMPLES_TEXT = "\n".join(
    f"TICKET: {t}\nCATEGORY: {c}" for t, c in FEW_SHOT_EXAMPLES
)

SYSTEM_FEW_SHOT = (
    "You are a CLASSIFY customer support triage assistant for Instamart. "
    "Classify support tickets into exactly one of: billing, delivery, technical, fraud. "
    "Respond ONLY with a JSON object: {\"category\": \"<one of the four>\", "
    "\"reason\": \"<one-line explanation>\"}.\n\n"
    f"EXAMPLES:\n{EXAMPLES_TEXT}"
)

# ─────────────────────────────────────────────────────────────
# 7.  CORE CLASSIFICATION FUNCTION
# ─────────────────────────────────────────────────────────────

def classify_ticket(ticket_text_redacted: str,
                    system_prompt: str,
                    token_budget_tracker: list,
                    budget_limit: int) -> dict:
    """
    Classify a single redacted ticket text.

    Args:
        ticket_text_redacted: PII-stripped ticket text
        system_prompt:        system message content
        token_budget_tracker: mutable [tokens_used] list (shared across calls)
        budget_limit:         stop calling if cumulative tokens would exceed this

    Returns dict with keys: category, parse_method, tokens_used, skipped_budget
    """
    # Estimate tokens for this call
    est = est_tokens(system_prompt) + est_tokens(ticket_text_redacted) + 20  # overhead
    if token_budget_tracker[0] + est > budget_limit:
        return {
            "category":       "skipped_budget",
            "parse_method":   "budget",
            "tokens_used":    0,
            "skipped_budget": True,
        }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": ticket_text_redacted},
    ]

    try:
        resp = call_with_retry(messages)
        content = resp.choices[0].message.content
        tokens  = resp.usage.total_tokens
        token_budget_tracker[0] += tokens
        cat, method = parse_response(content)
        return {
            "category":       cat,
            "parse_method":   method,
            "tokens_used":    tokens,
            "skipped_budget": False,
        }
    except TransientLLMError as exc:
        # All retries exhausted
        return {
            "category":       "error",
            "parse_method":   "retry_exhausted",
            "tokens_used":    0,
            "skipped_budget": False,
            "error":          str(exc),
        }


# ─────────────────────────────────────────────────────────────
# 8.  ZERO-SHOT RUN  (budget = 6,000 tokens)
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print(f"STEP 6 - Zero-shot run  (budget = {ZERO_SHOT_BUDGET:,} tokens)")
print(SEP)

reset()   # reset mock_llm call counters for clean run
budget_zs = [0]
zs_rows   = []

for _, row in processable.iterrows():
    result = classify_ticket(
        row["text_redacted"],
        SYSTEM_ZERO_SHOT,
        budget_zs,
        ZERO_SHOT_BUDGET,
    )
    zs_rows.append({
        "ticket_id":      row["ticket_id"],
        "customer_id":    row["customer_id"],
        "true_category":  row["true_category"],
        "pred_category":  result["category"],
        "parse_method":   result["parse_method"],
        "tokens_used":    result["tokens_used"],
        "cumulative_tokens": budget_zs[0],
        "skipped_budget": result["skipped_budget"],
        "prompt_mode":    "zero_shot",
    })

zs_df = pd.DataFrame(zs_rows)
path_zs = os.path.join(OUTPUTS, "triage_budget_demo.csv")
zs_df.to_csv(path_zs, index=False)

# Stats
processed_zs = zs_df[~zs_df["skipped_budget"]]
skipped_zs   = zs_df[zs_df["skipped_budget"]]
valid_zs      = processed_zs[processed_zs["pred_category"] != "error"]
correct_zs    = valid_zs[
    valid_zs["true_category"].isin(VALID_CATS) &
    (valid_zs["pred_category"] == valid_zs["true_category"])
]

print(f"  Total processable    : {len(processable):,}")
print(f"  Processed (in budget): {len(processed_zs):,}")
print(f"  Skipped (budget)     : {len(skipped_zs):,}")
print(f"  Tokens consumed      : {budget_zs[0]:,} / {ZERO_SHOT_BUDGET:,}")
print(f"  JSON parse success   : {(processed_zs['parse_method'] == 'json').sum()}")
print(f"  Keyword fallback     : {(processed_zs['parse_method'] == 'keyword').sum()}")
print(f"  Unknown / error      : {(processed_zs['parse_method'].isin(['unknown','retry_exhausted'])).sum()}")

eligible_true = valid_zs[valid_zs["true_category"].isin(VALID_CATS)]
if len(eligible_true):
    acc = len(correct_zs) / len(eligible_true)
    print(f"  Accuracy (valid cats): {acc:.2%}  ({len(correct_zs)}/{len(eligible_true)})")

print(f"\n  Prediction distribution (zero-shot):")
print(zs_df["pred_category"].value_counts().to_string())
print(f"\n  Saved -> {path_zs}")

# ─────────────────────────────────────────────────────────────
# 9.  FEW-SHOT RUN  (budget = 100,000 tokens) + SEGMENT ROUTING
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print(f"STEP 7 - Few-shot run  (budget = {FEW_SHOT_BUDGET:,} tokens)")
print(SEP)

# Segment-based priority routing rules:
#   Premium -> priority 1 (highest)
#   Loyal   -> priority 2
#   At-Risk -> priority 3
#   Lapsed / no segment -> priority 4
PRIORITY_MAP = {
    "Premium": 1,
    "Loyal":   2,
    "At-Risk": 3,
    "Lapsed":  4,
}
SLA_MAP = {
    "Premium": "1 hour",
    "Loyal":   "4 hours",
    "At-Risk": "8 hours",
    "Lapsed":  "24 hours",
}
ESCALATE_CATS = {"fraud"}   # always escalate fraud regardless of segment

# Merge segments into processable (left join so we keep all tickets)
proc_seg = processable.merge(segments, on="customer_id", how="left")
proc_seg["segment_label"] = proc_seg["segment_label"].fillna("Unknown")
proc_seg["priority"]      = proc_seg["segment_label"].map(PRIORITY_MAP).fillna(4).astype(int)
proc_seg["sla_target"]    = proc_seg["segment_label"].map(SLA_MAP).fillna("24 hours")

# Sort by priority so highest-value customers are classified first
proc_seg = proc_seg.sort_values(["priority", "ticket_id"]).reset_index(drop=True)

reset()   # fresh mock_llm counters
budget_fs = [0]
fs_rows   = []

for _, row in proc_seg.iterrows():
    result = classify_ticket(
        row["text_redacted"],
        SYSTEM_FEW_SHOT,
        budget_fs,
        FEW_SHOT_BUDGET,
    )
    pred_cat = result["category"]

    # ── Routing logic ──────────────────────────────────────────
    segment   = row["segment_label"]
    priority  = int(row["priority"])
    sla       = row["sla_target"]
    escalate  = pred_cat in ESCALATE_CATS   # fraud always escalates
    # Premium + fraud -> CRITICAL escalation
    if segment == "Premium" and escalate:
        route = "CRITICAL_ESCALATE"
    elif escalate:
        route = "ESCALATE_FRAUD"
    elif segment == "Premium":
        route = "PRIORITY_QUEUE"
    elif pred_cat == "unknown" or result["parse_method"] == "unknown":
        route = "MANUAL_REVIEW"
    elif result["skipped_budget"]:
        route = "SKIPPED_BUDGET"
    else:
        route = "STANDARD_QUEUE"

    fs_rows.append({
        "ticket_id":         row["ticket_id"],
        "customer_id":       row["customer_id"],
        "segment_label":     segment,
        "priority":          priority,
        "sla_target":        sla,
        "true_category":     row["true_category"],
        "pred_category":     pred_cat,
        "parse_method":      result["parse_method"],
        "tokens_used":       result["tokens_used"],
        "cumulative_tokens": budget_fs[0],
        "skipped_budget":    result["skipped_budget"],
        "route":             route,
        "escalate":          escalate,
        "prompt_mode":       "few_shot",
        "pii_email_count":   row["n_email_redacted"],
        "pii_phone_count":   row["n_phone_redacted"],
        "pii_order_count":   row["n_order_redacted"],
    })

fs_df = pd.DataFrame(fs_rows)
path_fs = os.path.join(OUTPUTS, "triage_results.csv")
fs_df.to_csv(path_fs, index=False)

# ── Stats ──────────────────────────────────────────────────────
processed_fs = fs_df[~fs_df["skipped_budget"]]
skipped_fs   = fs_df[fs_df["skipped_budget"]]
valid_fs     = processed_fs[processed_fs["pred_category"].isin(VALID_CATS)]
correct_fs   = valid_fs[
    valid_fs["true_category"].isin(VALID_CATS) &
    (valid_fs["pred_category"] == valid_fs["true_category"])
]

print(f"  Total processable    : {len(processable):,}")
print(f"  Processed (in budget): {len(processed_fs):,}")
print(f"  Skipped (budget)     : {len(skipped_fs):,}")
print(f"  Tokens consumed      : {budget_fs[0]:,} / {FEW_SHOT_BUDGET:,}")
print(f"  JSON parse success   : {(processed_fs['parse_method'] == 'json').sum()}")
print(f"  Keyword fallback     : {(processed_fs['parse_method'] == 'keyword').sum()}")
print(f"  Unknown / error      : {(processed_fs['parse_method'].isin(['unknown','retry_exhausted'])).sum()}")

eligible_true_fs = valid_fs[valid_fs["true_category"].isin(VALID_CATS)]
if len(eligible_true_fs):
    acc_fs = len(correct_fs) / len(eligible_true_fs)
    print(f"  Accuracy (valid cats): {acc_fs:.2%}  ({len(correct_fs)}/{len(eligible_true_fs)})")

print(f"\n  Route distribution:")
print(fs_df["route"].value_counts().to_string())

print(f"\n  Category distribution (few-shot):")
print(fs_df["pred_category"].value_counts().to_string())

# Per-category accuracy
print(f"\n  Per-category breakdown (true vs pred, few-shot):")
for cat in sorted(VALID_CATS):
    true_n = (fs_df["true_category"] == cat).sum()
    pred_n = (fs_df["pred_category"] == cat).sum()
    tp     = ((fs_df["true_category"] == cat) & (fs_df["pred_category"] == cat)).sum()
    print(f"    {cat:<12s}  true={true_n:>3}  pred={pred_n:>3}  TP={tp:>3}")

# Segment routing summary
print(f"\n  Segment routing summary:")
seg_summary = (
    fs_df.groupby("segment_label")
    .agg(
        tickets    = ("ticket_id", "count"),
        escalated  = ("escalate",  "sum"),
        avg_priority = ("priority", "mean"),
    )
    .sort_values("avg_priority")
)
print(seg_summary.to_string())

# Retry stats from mock_llm internal counters
print(f"\n  mock_llm stats  calls={mock_llm._state['calls']}  "
      f"ok={mock_llm._state['ok']}")

print(f"\n  Saved -> {path_fs}")

# ─────────────────────────────────────────────────────────────
# 10. VALIDATION ASSERTIONS
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 8 - Validation checkpoints")
print(SEP)

def check(name, cond, detail=""):
    mark = "[OK]" if cond else "[!!]"
    msg  = f"  {mark} [{'PASS' if cond else 'FAIL'}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    assert cond, f"Checkpoint failed: {name}  {detail}"

# Guardrail checks
check("all empty tickets skipped",
      all(tid not in processable["ticket_id"].values
          for tid in tickets[empty_mask]["ticket_id"]),
      f"empty tickets: {empty_mask.sum()}")

check("all over-token tickets skipped",
      all(tid not in processable["ticket_id"].values
          for tid in tickets[too_long_mask]["ticket_id"]),
      f"over-token tickets: {too_long_mask.sum()}")

# PII checks: no raw emails/phones in redacted text
check("no raw emails in redacted text",
      processable["text_redacted"].apply(
          lambda t: len(EMAIL_RE.findall(t)) == 0
      ).all())

check("no raw phones in redacted text",
      processable["text_redacted"].apply(
          lambda t: len(PHONE_RE.findall(t)) == 0
      ).all())

# Budget checks
check("zero-shot: tokens consumed <= budget",
      budget_zs[0] <= ZERO_SHOT_BUDGET,
      f"{budget_zs[0]:,} <= {ZERO_SHOT_BUDGET:,}")

check("few-shot: tokens consumed <= budget",
      budget_fs[0] <= FEW_SHOT_BUDGET,
      f"{budget_fs[0]:,} <= {FEW_SHOT_BUDGET:,}")

# Output files exist
check("triage_budget_demo.csv exists",
      os.path.exists(path_zs))

check("triage_results.csv exists",
      os.path.exists(path_fs))

# Predictions are always in valid set + budget sentinels
allowed_preds = VALID_CATS | {"unknown", "error", "skipped_budget"}
check("all pred_category values are from allowed set (zero-shot)",
      set(zs_df["pred_category"].unique()).issubset(allowed_preds),
      f"unexpected: {set(zs_df['pred_category'].unique()) - allowed_preds}")

check("all pred_category values are from allowed set (few-shot)",
      set(fs_df["pred_category"].unique()).issubset(allowed_preds),
      f"unexpected: {set(fs_df['pred_category'].unique()) - allowed_preds}")

# Route field present in few-shot results
check("route column present in triage_results.csv",
      "route" in pd.read_csv(path_fs).columns)

# Fraud tickets escalated in few-shot
fs_fraud_pred = fs_df[fs_df["pred_category"] == "fraud"]
if len(fs_fraud_pred):
    check("all fraud-predicted tickets are escalated",
          fs_fraud_pred["escalate"].all(),
          f"{fs_fraud_pred['escalate'].sum()}/{len(fs_fraud_pred)} escalated")

# Accuracy sanity: few-shot should be at least 60% on labelable tickets
if len(eligible_true_fs) > 0:
    check("few-shot accuracy >= 60% on labelled tickets",
          acc_fs >= 0.60,
          f"{acc_fs:.2%}")

print()
print(SEP)
print("ALL CHECKPOINTS PASSED  [OK]")
print(SEP)
print()
print("Part C complete.  Files written:")
print(f"  {path_zs}  ({len(zs_df):,} rows)")
print(f"  {path_fs}  ({len(fs_df):,} rows)")
