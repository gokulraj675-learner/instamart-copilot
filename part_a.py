"""
part_a.py -- Instamart Capstone Part A
=======================================
1. Audit raw CSVs and print exact stats
2. Clean customers.csv  -> data/clean/customers.csv
3. Clean orders.csv     -> data/clean/orders.csv
4. Persist to           -> data/instamart.db
5. Validate checkpoints with assertions

Allowed libraries: standard Python, pandas, numpy, scikit-learn, joblib, sqlite3
"""

import os
import sqlite3
import sys

# Force UTF-8 output so Unicode characters print on Windows PowerShell / cmd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────────────────────
RAW   = os.path.join("data", "raw")
CLEAN = os.path.join("data", "clean")
DB    = os.path.join("data", "instamart.db")
os.makedirs(CLEAN, exist_ok=True)

SEP = "=" * 65

# ─────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────────────────────
print(SEP)
print("STEP 1 · Loading raw CSVs")
print(SEP)

cust_raw   = pd.read_csv(os.path.join(RAW, "customers.csv"))
orders_raw = pd.read_csv(os.path.join(RAW, "orders.csv"))
products   = pd.read_csv(os.path.join(RAW, "products.csv"))

print(f"  customers  : {cust_raw.shape[0]:>6,} rows × {cust_raw.shape[1]} cols")
print(f"  orders     : {orders_raw.shape[0]:>6,} rows × {orders_raw.shape[1]} cols")
print(f"  products   : {products.shape[0]:>6,} rows × {products.shape[1]} cols")

# ─────────────────────────────────────────────────────────────
# 2. AUDIT
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 2 · Data Audit")
print(SEP)

# ── 2a. Exact duplicates ──────────────────────────────────────
print("\n[2a] Duplicate rows")
cust_exact_dups   = int(cust_raw.duplicated().sum())
orders_exact_dups = int(orders_raw.duplicated().sum())
print(f"  customers  exact duplicate rows : {cust_exact_dups}")
print(f"  orders     exact duplicate rows : {orders_exact_dups}")

# Partial duplicates in orders: same order_id, different status/timestamp
orders_partial_dups = int(
    orders_raw.duplicated(subset=["order_id"], keep=False).sum()
    - orders_raw.duplicated(subset=["order_id", "status", "order_ts"], keep=False).sum()
)
print(f"  orders     same order_id, multiple status rows : "
      f"{orders_raw[orders_raw.duplicated(subset=['order_id'], keep=False)].shape[0]}")

# ── 2b. Missing values ────────────────────────────────────────
print("\n[2b] Missing / null values")
print("  customers:")
for col in cust_raw.columns:
    n = int(cust_raw[col].isna().sum())
    if n or col in ("age", "gender"):
        print(f"    {col:<20s}: {n:>5} missing")

# orders: age col not present; focus on order_amount and delivery_minutes
print("  orders:")
for col in orders_raw.columns:
    n = int(orders_raw[col].isna().sum())
    if n or col in ("order_amount", "delivery_minutes"):
        print(f"    {col:<22s}: {n:>5} missing")

# orders: non-parseable amounts (N/A or 'error')
orders_raw_amounts = orders_raw["order_amount"].copy()
orders_bad_amount_mask = pd.to_numeric(orders_raw_amounts, errors="coerce").isna() & orders_raw_amounts.notna()
print(f"  orders     non-numeric order_amount (planted 'N/A'/'error') : {orders_bad_amount_mask.sum()}")

# ── 2c. Bad ages in customers ────────────────────────────────
print("\n[2c] Age anomalies in customers")
cust_age_num = pd.to_numeric(cust_raw["age"], errors="coerce")

age_missing     = int(cust_age_num.isna().sum())          # blank or non-numeric
age_impossible  = int(((cust_age_num < 18) | (cust_age_num > 100)).sum())
age_ok          = int(((cust_age_num >= 18) & (cust_age_num <= 100)).sum())

print(f"  Missing / blank age             : {age_missing}")
print(f"  Impossible ages (< 18 or > 100) : {age_impossible}")
print(f"    Detail -> impossible values : "
      f"{sorted(cust_age_num[(cust_age_num < 18) | (cust_age_num > 100)].unique().tolist())}")
print(f"  Valid ages (18-100)             : {age_ok}")

# ── 2d. City spelling variants ────────────────────────────────
print("\n[2d] City spelling variants in customers (raw counts)")
city_variants = (
    cust_raw["city"]
    .str.strip()
    .str.title()
    .value_counts()
    .rename_axis("city_normalized")
    .reset_index(name="count_after_normalise")
)
raw_city_vc = cust_raw["city"].value_counts()
print("  Raw unique city strings found:")
for v, cnt in raw_city_vc.items():
    print(f"    {repr(v):<30s} : {cnt}")

# ─────────────────────────────────────────────────────────────
# 3. CLEAN customers.csv
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 3 · Cleaning customers.csv")
print(SEP)

cust = cust_raw.copy()

# 3a. Standardise city: strip whitespace, title-case
cust["city"] = cust["city"].str.strip().str.title()
print(f"  [3a] Cities standardised. Unique cities now: {sorted(cust['city'].unique())}")

# 3b. Standardise gender → 'Male' / 'Female' / NaN
GENDER_MAP = {
    "male": "Male", "m": "Male", "MALE": "Male", "Male": "Male",
    "female": "Female", "f": "Female", "FEMALE": "Female", "Female": "Female",
}
cust["gender"] = cust["gender"].map(lambda x: GENDER_MAP.get(str(x).strip(), np.nan) if pd.notna(x) and str(x).strip() != "" else np.nan)
print(f"  [3b] Gender values. Unique: {sorted(cust['gender'].dropna().unique())}")
print(f"       Missing gender (NaN)  : {cust['gender'].isna().sum()}")

# 3c. Fix ages
#     • Convert to numeric (non-parseable → NaN)
#     • Clamp impossible ages (< 18 or > 100) → NaN
#     • Fill NaN ages with city-specific median of valid ages
cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
cust.loc[(cust["age"] < 18) | (cust["age"] > 100), "age"] = np.nan

# City median from valid ages only
city_age_median = cust.groupby("city")["age"].median()
print(f"  [3c] City age medians (valid rows):\n{city_age_median.to_string()}")

before_fill = cust["age"].isna().sum()
cust["age"] = cust.apply(
    lambda r: city_age_median[r["city"]] if pd.isna(r["age"]) else r["age"],
    axis=1,
)
after_fill = cust["age"].isna().sum()
print(f"       Ages filled (was NaN) : {before_fill - after_fill}")
print(f"       Ages still missing    : {after_fill}")

# Round age to integer (medians can be .5)
cust["age"] = cust["age"].round().astype(int)

cust_out_path = os.path.join(CLEAN, "customers.csv")
cust.to_csv(cust_out_path, index=False)
print(f"  Saved -> {cust_out_path}  ({cust.shape[0]:,} rows)")

# ─────────────────────────────────────────────────────────────
# 4. CLEAN orders.csv
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 4 · Cleaning orders.csv")
print(SEP)

orders = orders_raw.copy()

# 4a. Remove exact duplicate rows
before = len(orders)
orders = orders.drop_duplicates()
print(f"  [4a] Exact duplicate rows removed  : {before - len(orders)}  ({len(orders):,} remain)")

# 4b. For same order_id with multiple status rows, keep latest status row
#     (sort by order_ts desc so first occurrence after dedup is the latest)
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders = orders.sort_values("order_ts", ascending=False)
before = len(orders)
orders = orders.drop_duplicates(subset=["order_id"], keep="first")
print(f"  [4b] Partial-duplicate rows removed: {before - len(orders)}  ({len(orders):,} remain)")
orders = orders.sort_values("order_ts").reset_index(drop=True)

# 4c. Fix order_amount
#     • Non-numeric strings (N/A, error) → NaN
#     • Recalculate missing amounts using price * qty * (1 - discount_pct/100)
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")

# Join product prices
price_map = products.set_index("product_id")["price_inr"]
orders["_unit_price"] = orders["product_id"].map(price_map)

calc_mask = orders["order_amount"].isna()
orders.loc[calc_mask, "order_amount"] = (
    orders.loc[calc_mask, "_unit_price"]
    * orders.loc[calc_mask, "quantity"]
    * (1 - orders.loc[calc_mask, "discount_pct"] / 100)
).round(2)
print(f"  [4c] order_amount recalculated for : {calc_mask.sum()} rows")
orders = orders.drop(columns=["_unit_price"])

# 4d. Fix delivery_minutes
#     • Convert to numeric (non-numeric / blank → NaN)
#     • Negative values (planted -10) → NaN
#     • NaN delivery_minutes for completed orders → category median of valid, positive values
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
neg_mask = orders["delivery_minutes"] < 0
print(f"  [4d] Negative delivery_minutes      : {neg_mask.sum()}  (-> NaN)")
orders.loc[neg_mask, "delivery_minutes"] = np.nan

# Join category to orders for median calculation
cat_map = products.set_index("product_id")["category"]
orders["_category"] = orders["product_id"].map(cat_map)

completed = orders["status"] == "completed"
cat_deliv_median = (
    orders[completed & orders["delivery_minutes"].notna()]
    .groupby("_category")["delivery_minutes"]
    .median()
)
print(f"  [4d] Category delivery medians (completed orders with valid minutes):")
print(cat_deliv_median.to_string())

missing_deliv_mask = completed & orders["delivery_minutes"].isna()
filled_count = missing_deliv_mask.sum()
orders.loc[missing_deliv_mask, "delivery_minutes"] = orders.loc[missing_deliv_mask, "_category"].map(cat_deliv_median)
print(f"  [4d] delivery_minutes filled        : {filled_count} rows")

orders = orders.drop(columns=["_category"])
orders["delivery_minutes"] = orders["delivery_minutes"].where(orders["delivery_minutes"].notna(), other=np.nan)

orders_out_path = os.path.join(CLEAN, "orders.csv")
orders.to_csv(orders_out_path, index=False)
print(f"  Saved -> {orders_out_path}  ({orders.shape[0]:,} rows)")

# ─────────────────────────────────────────────────────────────
# 5. SAVE TO SQLite
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 5 - Writing to SQLite -> data/instamart.db")
print(SEP)

conn = sqlite3.connect(DB)

# Use pandas to_sql with replace so re-runs are idempotent
cust.to_sql("customers", conn, if_exists="replace", index=False)
orders.to_sql("orders",   conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)

conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_orders_customer
    ON orders(customer_id)
""")
conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_orders_product
    ON orders(product_id)
""")
conn.commit()

# Verify row counts
for tbl in ("customers", "orders", "products"):
    n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl:<12s}: {n:>6,} rows in DB")

conn.close()
print(f"  DB saved -> {DB}")

# ─────────────────────────────────────────────────────────────
# 6. VALIDATION (assertions)
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 6 · Running validation checkpoints")
print(SEP)

def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    mark  = "[OK]" if condition else "[!!]"
    msg = f"  {mark} [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    assert condition, f"Checkpoint failed: {name}  {detail}"

# ── Customers ────────────────────────────────────────────────
cust_clean = pd.read_csv(os.path.join(CLEAN, "customers.csv"))

check("customers: no exact duplicates",
      cust_clean.duplicated().sum() == 0,
      f"found {cust_clean.duplicated().sum()}")

check("customers: no missing age",
      cust_clean["age"].isna().sum() == 0,
      f"found {cust_clean['age'].isna().sum()}")

check("customers: all ages in [18, 100]",
      ((cust_clean["age"] >= 18) & (cust_clean["age"] <= 100)).all(),
      f"bad: {cust_clean[(cust_clean['age'] < 18) | (cust_clean['age'] > 100)].shape[0]}")

VALID_CITIES = {"Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune"}
check("customers: all cities title-cased and valid",
      set(cust_clean["city"].unique()).issubset(VALID_CITIES),
      f"unexpected: {set(cust_clean['city'].unique()) - VALID_CITIES}")

check("customers: gender is Male/Female/NaN only",
      cust_clean["gender"].dropna().isin(["Male", "Female"]).all(),
      f"bad count: {(~cust_clean['gender'].dropna().isin(['Male','Female'])).sum()}")

check("customers: original row count preserved",
      len(cust_clean) == len(cust_raw),
      f"expected {len(cust_raw)}, got {len(cust_clean)}")

# ── Orders ────────────────────────────────────────────────────
orders_clean = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
orders_clean["order_ts"] = pd.to_datetime(orders_clean["order_ts"])

check("orders: no exact duplicates",
      orders_clean.duplicated().sum() == 0,
      f"found {orders_clean.duplicated().sum()}")

check("orders: no duplicate order_id",
      orders_clean.duplicated(subset=["order_id"]).sum() == 0,
      f"found {orders_clean.duplicated(subset=['order_id']).sum()}")

check("orders: order_amount fully numeric and non-null",
      orders_clean["order_amount"].isna().sum() == 0 and
      pd.to_numeric(orders_clean["order_amount"], errors="coerce").isna().sum() == 0,
      "")

check("orders: no negative delivery_minutes",
      (orders_clean["delivery_minutes"].dropna() >= 0).all(),
      f"negatives: {(orders_clean['delivery_minutes'].dropna() < 0).sum()}")

check("orders: no missing delivery_minutes for completed orders",
      orders_clean[orders_clean["status"] == "completed"]["delivery_minutes"].isna().sum() == 0,
      f"still missing: {orders_clean[orders_clean['status'] == 'completed']['delivery_minutes'].isna().sum()}")

check("orders: count is less than raw (duplicates removed)",
      len(orders_clean) < len(orders_raw),
      f"clean={len(orders_clean):,}  raw={len(orders_raw):,}")

# ── SQLite ────────────────────────────────────────────────────
conn = sqlite3.connect(DB)

n_cust_db = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
check("sqlite: customers row count matches clean CSV",
      n_cust_db == len(cust_clean),
      f"db={n_cust_db}  csv={len(cust_clean)}")

n_ord_db = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
check("sqlite: orders row count matches clean CSV",
      n_ord_db == len(orders_clean),
      f"db={n_ord_db}  csv={len(orders_clean)}")

n_prod_db = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
check("sqlite: products row count matches raw CSV",
      n_prod_db == len(products),
      f"db={n_prod_db}  raw={len(products)}")

# Quick SQL sanity: join works
joined = conn.execute("""
    SELECT COUNT(*) FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products  p ON o.product_id  = p.product_id
""").fetchone()[0]
check("sqlite: three-table join returns rows > 0",
      joined > 0,
      f"joined rows = {joined}")

conn.close()

print()
print(SEP)
print("ALL CHECKPOINTS PASSED  [OK]")
print(SEP)
print()
print("Part A complete.  Files written:")
print(f"  {os.path.join(CLEAN, 'customers.csv')}")
print(f"  {os.path.join(CLEAN, 'orders.csv')}")
print(f"  {DB}")
print(f"  sql\\queries.sql  (see separate file)")
