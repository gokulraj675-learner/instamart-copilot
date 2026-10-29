-- =============================================================
-- sql/queries.sql  —  Instamart Capstone Part A
-- Database: data/instamart.db
-- Tables  : customers, orders, products
-- =============================================================

-- ─────────────────────────────────────────────────────────────
-- Q1.  WHERE + HAVING
--      High-value cities: cities where avg completed order
--      amount > 270 INR and total completed orders >= 1000.
-- ─────────────────────────────────────────────────────────────
SELECT
    c.city,
    COUNT(o.order_id)                    AS total_completed_orders,
    ROUND(AVG(o.order_amount), 2)        AS avg_order_amount_inr
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE  o.status = 'completed'
GROUP  BY c.city
HAVING avg_order_amount_inr > 270
   AND total_completed_orders >= 1000
ORDER  BY avg_order_amount_inr DESC;


-- ─────────────────────────────────────────────────────────────
-- Q2.  INNER JOIN  (three-table join)
--      Full order detail: customer name + city, product name,
--      category, quantity, discount, and final amount for
--      all completed orders.
-- ─────────────────────────────────────────────────────────────
SELECT
    o.order_id,
    c.customer_id,
    c.city,
    p.product_name,
    p.category,
    o.quantity,
    o.discount_pct,
    o.order_amount,
    o.delivery_minutes,
    o.order_ts
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id
WHERE  o.status = 'completed'
ORDER  BY o.order_ts DESC
LIMIT  100;


-- ─────────────────────────────────────────────────────────────
-- Q3.  SCALAR SUBQUERY
--      Orders whose amount is above the overall average order
--      amount for completed orders (one scalar from subquery).
-- ─────────────────────────────────────────────────────────────
SELECT
    o.order_id,
    c.city,
    p.category,
    o.order_amount,
    ROUND(
        (SELECT AVG(order_amount) FROM orders WHERE status = 'completed')
    , 2)                                 AS overall_avg_amount,
    ROUND(
        o.order_amount -
        (SELECT AVG(order_amount) FROM orders WHERE status = 'completed')
    , 2)                                 AS diff_from_avg
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id
WHERE  o.status = 'completed'
  AND  o.order_amount > (
           SELECT AVG(order_amount) FROM orders WHERE status = 'completed'
       )
ORDER  BY o.order_amount DESC
LIMIT  50;


-- ─────────────────────────────────────────────────────────────
-- Q4.  LEFT JOIN
--      Customers who have NEVER placed a completed order
--      (shows NULLs on the right side of the LEFT JOIN).
-- ─────────────────────────────────────────────────────────────
SELECT
    c.customer_id,
    c.city,
    c.age,
    c.gender,
    c.signup_date,
    o.order_id      AS first_completed_order   -- NULL if none
FROM customers c
LEFT JOIN orders o
    ON  o.customer_id = c.customer_id
    AND o.status      = 'completed'
WHERE  o.order_id IS NULL
ORDER  BY c.signup_date;


-- ─────────────────────────────────────────────────────────────
-- Q5.  CTE  (Common Table Expression)
--      Identify "late delivery" orders (delivery > 45 min)
--      and compute per-category late rate and avg late minutes.
-- ─────────────────────────────────────────────────────────────
WITH completed_with_delivery AS (
    -- Base: completed orders that have a valid delivery time
    SELECT
        o.order_id,
        p.category,
        o.delivery_minutes,
        CASE WHEN o.delivery_minutes > 45 THEN 1 ELSE 0 END AS is_late
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
    WHERE  o.status = 'completed'
      AND  o.delivery_minutes IS NOT NULL
),
category_stats AS (
    SELECT
        category,
        COUNT(*)                            AS total_orders,
        SUM(is_late)                        AS late_orders,
        ROUND(100.0 * SUM(is_late) / COUNT(*), 2) AS late_pct,
        ROUND(AVG(delivery_minutes), 2)     AS avg_delivery_min,
        ROUND(AVG(CASE WHEN is_late = 1 THEN delivery_minutes END), 2)
                                            AS avg_late_minutes
    FROM completed_with_delivery
    GROUP BY category
)
SELECT *
FROM   category_stats
ORDER  BY late_pct DESC;


-- ─────────────────────────────────────────────────────────────
-- Q6.  WINDOW FUNCTIONS
--      For each customer, rank their orders by amount (desc)
--      and compute each order's percentile within the city.
--      Also show dense rank of customers by total spend per city.
-- ─────────────────────────────────────────────────────────────
WITH customer_order_totals AS (
    SELECT
        o.order_id,
        o.customer_id,
        c.city,
        o.order_amount,
        o.order_ts,
        -- Rank order amount within the customer's own history
        RANK() OVER (
            PARTITION BY o.customer_id
            ORDER BY o.order_amount DESC
        )                                   AS order_rank_by_amount,
        -- Percentile of this order within the city
        ROUND(
            100.0 * PERCENT_RANK() OVER (
                PARTITION BY c.city
                ORDER BY o.order_amount
            ), 2
        )                                   AS city_amount_pct_rank,
        -- Running count of orders for the customer over time
        ROW_NUMBER() OVER (
            PARTITION BY o.customer_id
            ORDER BY o.order_ts
        )                                   AS customer_order_seq
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE  o.status = 'completed'
)
SELECT *
FROM   customer_order_totals
ORDER  BY city, order_amount DESC
LIMIT  200;


-- ─────────────────────────────────────────────────────────────
-- Q7.  RUNNING TOTAL  (cumulative revenue per city over time)
--      Uses a window function with ROWS BETWEEN UNBOUNDED
--      PRECEDING AND CURRENT ROW ordered by order date.
-- ─────────────────────────────────────────────────────────────
WITH daily_city_revenue AS (
    SELECT
        c.city,
        DATE(o.order_ts)                    AS order_date,
        ROUND(SUM(o.order_amount), 2)       AS daily_revenue
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE  o.status = 'completed'
    GROUP  BY c.city, DATE(o.order_ts)
)
SELECT
    city,
    order_date,
    daily_revenue,
    ROUND(
        SUM(daily_revenue) OVER (
            PARTITION BY city
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 2
    )                                       AS cumulative_revenue
FROM   daily_city_revenue
ORDER  BY city, order_date;
