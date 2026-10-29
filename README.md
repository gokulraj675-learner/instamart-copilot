# Instamart Copilot

## Overview

This project is a four-part data and AI pipeline built on a synthetic Indian quick-commerce dataset called Instamart. The dataset covers 1,200 customers, roughly 13,000 orders, and 40 products across six cities: Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, and Pune.

Part A audits and cleans the raw data then loads it into SQLite. Part B segments customers with K-Means RFM clustering and trains a lapse-prediction classifier. Part C classifies incoming support tickets using a mock LLM under strict token budgets. Part D builds a policy-grounded ReAct agent that resolves refund requests end to end.

All four scripts run offline. No external API key or network connection is required. The only third-party dependencies beyond the Python standard library are pandas, numpy, scikit-learn, and the local mock_llm module bundled in the repository.


## Reproduce

Clone or download the repository so that the following layout is present at the working directory:

    data/raw/customers.csv
    data/raw/orders.csv
    data/raw/products.csv
    data/raw/tickets.csv
    data/raw/agent_scenarios.csv
    data/policies/   (five .txt files)
    mock_llm.py
    part_a.py
    part_b.py
    part_c.py
    part_d.py

Create and activate a virtual environment, then install the required packages:

    python -m venv venv
    venv\Scripts\activate          # Windows
    # source venv/bin/activate     # macOS / Linux
    pip install pandas numpy scikit-learn

Run the four parts in order. Each script is fully self-contained and writes its own outputs:

    python part_a.py
    python part_b.py
    python part_c.py
    python part_d.py

Output files land in the outputs/ directory. SQLite database is written to data/instamart.db. Checkpoint files for Part D are written to state/.

To exercise the checkpoint resume feature, run part_d.py a second time. All seven scenarios will be skipped and loaded from state/ without re-invoking the agent.


## Part A

Part A loads three raw CSV files, runs a structured data audit, cleans every table, and writes the results to data/clean/ and data/instamart.db.

Raw input sizes were 1,200 rows across 5 columns for customers, 13,084 rows across 9 columns for orders, and 40 rows across 4 columns for products.

The data audit produced the following findings.

Duplicate rows: the customers table had zero exact duplicate rows. The orders table had 119 exact duplicate rows and an additional 1,547 rows sharing an order_id with a conflicting status field, indicating partial-duplicate records that required deduplication by keeping only the most recent status per order.

Missing values: the customers table was missing 61 age values and 30 gender values. The orders table was missing 53 order_amount values and 2,541 delivery_minutes values. An additional 67 order_amount entries contained non-numeric planted strings such as N/A or error.

Age anomalies: 61 age values were null and 6 were biologically impossible, taking the values -4, 0, 150, 200, 280, and 999. The 1,133 remaining ages were valid. Nulls were imputed using city-level medians: Bengaluru 42.0, Chennai 39.0, Delhi 43.0, Hyderabad 37.5, Mumbai 38.5, and Pune 39.0.

City spelling variants: each of the six city names appeared in up to eight orthographic variants combining different cases and leading or trailing whitespace. The raw data contained 24 distinct city strings that were all normalised to six canonical title-cased values.

Cleaning steps applied: cities were standardised, gender nulls were retained as NaN (not imputed), 119 exact duplicate orders were removed leaving 12,965 rows, a further 656 partial duplicates were resolved leaving 12,309 rows, order amounts were recalculated from quantity and product price for 112 rows where the stored value was wrong or missing, 105 negative delivery_minutes values were set to NaN, and 774 missing delivery_minutes were filled using category-level medians computed only from completed orders. The category medians were Dairy 25.0, Electronics 25.0, Grocery 25.0, Personal Care 25.5, and Snacks 25.0.

All 16 validation checkpoints passed. The final clean tables contain 1,200 customer rows, 12,309 order rows, and 40 product rows. The SQLite database at data/instamart.db mirrors these counts exactly and the three-table join returns 12,309 rows.


## Part B

Part B computes RFM features from the clean orders table, clusters customers with K-Means, labels the clusters as business segments, and trains a lapse-prediction classifier.

RFM was computed at a snapshot date of 2025-10-01. Of 1,200 customers, 1,175 had at least one completed order and were included. Mean recency was 33.56 days, mean order frequency was 9.46 orders, and mean monetary value was INR 2,552.06.

K-Means was evaluated for K from 2 to 8 on scaled RFM features. The inertia and silhouette scores were:

K = 2: inertia 1820.8, silhouette 0.4650
K = 3: inertia 1268.2, silhouette 0.4452
K = 4: inertia 743.1, silhouette 0.5255
K = 5: inertia 617.4, silhouette 0.4805
K = 6: inertia 503.2, silhouette 0.4819
K = 7: inertia 408.5, silhouette 0.4163
K = 8: inertia 366.5, silhouette 0.4123

K = 4 produced the highest silhouette score of 0.5255 and was chosen as the final model. PCA reduced the feature space to two components explaining 63.4 percent and 21.8 percent of variance respectively, for a cumulative total of 85.2 percent.

The four clusters were mapped to business labels based on their unscaled centroid profiles. The Premium segment contains 419 customers with a mean recency of 8.77 days, mean frequency of 19.40 orders, and mean monetary value of INR 3,410.02. The At-Risk segment contains 436 customers with a mean recency of 28.86 days, mean frequency of 3.77 orders, and mean monetary value of INR 1,057.62. The Loyal segment contains 138 customers with a mean recency of 24.12 days, mean frequency of 6.77 orders, and mean monetary value of INR 7,129.70. The Lapsed segment contains 182 customers with a mean recency of 109.04 days, mean frequency of 2.20 orders, and mean monetary value of INR 685.99.

For lapse prediction, a binary label was defined using a cutoff date of 2025-08-01. A customer was considered lapsed if they had no completed orders after the cutoff. Of 1,171 customers with pre-cutoff activity, 450 were labelled lapsed (38.4 percent) and 721 were active. Sixteen behavioural features were engineered covering recency, frequency, monetary aggregates, cancel rate, category diversity, inter-order gap, late delivery rate, preferred category, and days since signup.

A stratified 80/20 train-test split yielded 936 training samples and 235 test samples, both with a lapse rate near 38 percent. Two classifiers were evaluated by GridSearchCV with cross-validated ROC-AUC scoring.

Logistic Regression with C=10.0 and L2 penalty achieved a cross-validation ROC-AUC of 0.7766 and a test ROC-AUC of 0.8053 with a test F1 of 0.7005. Random Forest with 200 estimators and max depth 10 achieved a cross-validation ROC-AUC of 0.7898 and a test ROC-AUC of 0.7934 with a test F1 of 0.7005.

Random over-sampling was applied to the training set to address the 38/62 class imbalance. After over-sampling, Logistic Regression reached a test ROC-AUC of 0.8087 and Recall of 0.8667 for the Lapsed class. Random Forest after over-sampling achieved a test ROC-AUC of 0.8047. The best model was Logistic Regression with over-sampling at ROC-AUC 0.8087 and was saved to outputs/lapse_model.joblib.

The top three features by Random Forest importance were monetary_sum at 0.1323, frequency at 0.1210, and total_orders at 0.1095.

A retention priority list of the 419 Premium customers was written to outputs/retention_list.csv. Only 1 Premium customer had a predicted lapse probability above 0.5. The average lapse probability across the Premium segment was 0.117.


## Part C

Part C classifies 164 support tickets into four categories: billing, delivery, technical, and fraud. It uses mock_llm.chat() in CLASSIFY mode under a strict token budget and applies PII redaction, retry logic, and segment-based routing.

Input guardrails removed 4 empty tickets and 4 tickets exceeding 400 tokens (ticket IDs T0157 through T0164), leaving 156 processable tickets. The true category distribution in those 156 tickets was 43 delivery, 39 fraud, 39 billing, and 39 technical, with 4 uncategorised.

PII redaction replaced 278 sensitive tokens in total: 55 email addresses, 91 phone numbers, and 132 order identifiers.

The retry wrapper uses a maximum of 3 retries with exponential backoff following the formula wait = 0.5 times 2 to the power of attempt plus uniform jitter between 0 and 0.5 seconds. The mock LLM raises a transient error on every ninth call.

Zero-shot run at a 6,000-token budget: 72 of 156 tickets were processed before the budget was exhausted, consuming 5,932 of 6,000 tokens. The remaining 84 tickets were skipped. Of the processed tickets, 63 responses were parsed as valid JSON, 5 were recovered by keyword fallback, and 4 were unresolvable. Accuracy on tickets with a valid true label was 79.17 percent (57 out of 72 eligible).

Few-shot run at a 100,000-token budget: all 156 tickets were processed, consuming 28,744 tokens. Of these, 135 responses were valid JSON, 12 were recovered by keyword fallback, and 9 were unresolvable unknowns. Accuracy on the 147 tickets with a known true label was 100.00 percent. The route distribution was 68 tickets to STANDARD_QUEUE, 47 to PRIORITY_QUEUE for Premium customers, 20 to ESCALATE_FRAUD, 15 to CRITICAL_ESCALATE for Premium customers flagged as fraud, and 6 to MANUAL_REVIEW. All 35 fraud-predicted tickets were escalated as required.


## Part D

Part D builds a policy RAG retriever and a ReAct refund agent grounded in that retriever. It runs the agent against seven pre-defined scenarios and writes every run to JSONL files with a per-scenario checkpoint.

The five policy files in data/policies/ were chunked into 40-word windows with a 10-word overlap, producing 15 chunks in total: 3 from delivery_policy.txt, 3 from escalation_policy.txt, 3 from payment_policy.txt, 2 from privacy_policy.txt, and 4 from refund_policy.txt. The chunks were vectorised with TfidfVectorizer using unigrams and bigrams, producing a vocabulary of 368 terms. The retrieve function returns the top-k chunks by cosine similarity. For the query "refund late delivery", the top result was chunk 1 of refund_policy.txt with a score of 0.3017.

The TOOL_MAP exposes five tools. get_order performs an O(1) lookup in the in-memory orders dictionary. search_policy calls the TF-IDF retriever with k=3. calculate_refund applies the refund rules: delivery over 45 minutes returns 100 percent of the order amount; delivery at or below 45 minutes returns zero. issue_refund records an ISSUED entry in the ledger sink. escalate_to_human records an ESCALATED entry in the ledger sink for refunds above INR 1,000.

The run_agent loop enforces three safeguards per scenario: a maximum of 6 iterations, a 6,000-token budget measured across all messages in the conversation, and a 10-second wall-clock time limit. Any safeguard that fires produces a labelled fallback outcome of TIMEOUT, TOKEN_LIMIT, or MAX_ITER respectively.

The seven scenarios from agent_scenarios.csv were all resolved correctly, giving an outcome match rate of 7 out of 7 (100 percent).

Scenario S1: customer C0007, order ORD-100048. Delivery took 65 minutes. The agent called get_order, then search_policy, then calculate_refund, then issue_refund, and terminated with REFUND after 5 iterations consuming 1,346 tokens. Ledger entry: ISSUED INR 128.25.

Scenario S2: customer C0008, order ORD-100067. Delivery exceeded 45 minutes. The agent followed the same four-tool path and terminated with REFUND after 5 iterations consuming 1,346 tokens. Ledger entry: ISSUED INR 135.00. Expected outcome: REFUND. Match: yes.

Scenario S3: customer C0016, order ORD-100145. Order amount was INR 1,999.00, above the INR 1,000 auto-approval threshold. The agent called calculate_refund which returned 1,999.00, then called escalate_to_human, and terminated with ESCALATE after 5 iterations consuming 1,353 tokens. Ledger entry: ESCALATED INR 1,999.00. Expected outcome: ESCALATE. Match: yes.

Scenario S4: customer C0001, order ORD-100001. The order was delivered within 45 minutes. The agent terminated with DENIED after 4 iterations consuming 988 tokens. No ledger entry. Expected outcome: DENIED. Match: yes.

Scenario S5: customer C0003, order ORD-100025. The order was not in a completed status. The agent terminated with DENIED after 3 iterations consuming 646 tokens. No ledger entry. Expected outcome: DENIED. Match: yes.

Scenario S6: customer C0001, order ORD-999999. The order did not exist in the database. get_order returned an error. The agent terminated with NOT_FOUND after 2 iterations consuming 146 tokens. No ledger entry. Expected outcome: NOT_FOUND. Match: yes.

Scenario S7: customer C0007, message prefixed with [LOOP]. The mock LLM enters a loop calling search_policy on every iteration. The agent exhausted all 6 iterations consuming 4,428 tokens and terminated with MAX_ITER, which maps to the expected FALLBACK outcome. No ledger entry. Expected outcome: FALLBACK. Match: yes.

Total ledger: 2 ISSUED entries totalling INR 263.25 and 1 ESCALATED entry totalling INR 1,999.00. Outputs were written to outputs/agent_runs.jsonl and outputs/refund_ledger.jsonl. Seven checkpoint files were created in the state/ directory. On a second run, all seven scenarios are resumed from their checkpoints without re-invoking the agent.


## Limitations and Next Steps

The data is fully synthetic. All customer profiles, orders, and support tickets were generated by generate_data.py using fixed random seeds. Relationships such as the city distribution, refund rates, and ticket category frequencies do not reflect real operational data. Any model trained here should be re-validated on live data before deployment.

The Premium segment label leaks into Part C routing. The few-shot triage pipeline joins on customer_id to retrieve the segment label before calling the LLM and uses it to set SLA targets and escalation routes. This means the label produced by Part B is consumed at inference time in Part C without any holdout safeguard. If the segment model is retrained on the same orders data that drives the tickets, the performance figures in Part C will be optimistically biased. In production the segment assignment must be treated as a feature that could drift and should be evaluated on a time-based out-of-sample split.

The TF-IDF retriever in Part D is bag-of-words and cannot handle paraphrase or semantic similarity. A policy that says "full reimbursement" would not match a query containing "100 percent refund". Replacing TF-IDF with a dense retriever such as a sentence transformer would improve recall over longer or more varied policy corpora.

The mock LLM is deterministic and does not simulate real latency, hallucination, or prompt-sensitivity. The ReAct loop was designed and tested against mock_llm behaviour. A real LLM may produce tool calls with unexpected argument keys, refuse to call a tool, or return free-form text that the _classify_terminal function cannot map to a known outcome. The keyword matching in _classify_terminal would need to be replaced with a more robust parser or a structured output schema.

The lapse model was trained on a six-month window ending 2025-09-30 with a fixed cutoff of 2025-08-01. The definition of lapse is therefore window-length-dependent. A customer who took a two-month holiday would be incorrectly labelled as lapsed. Adding a longer observation window and testing multiple cutoff horizons would reduce this fragility.

The refund agent currently has no idempotency guard. If the agent is re-run for a scenario whose checkpoint was deleted, it will re-invoke issue_refund or escalate_to_human and append a duplicate entry to refund_ledger.jsonl. A production system would need to check the ledger for an existing record keyed on order_id before issuing a second refund.
