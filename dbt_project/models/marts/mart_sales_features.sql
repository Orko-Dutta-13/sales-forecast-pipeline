WITH train AS (
    SELECT * FROM {{ ref('stg_train') }}
),

stores AS (
    SELECT * FROM {{ ref('stg_stores') }}
),

oil AS (
    SELECT * FROM {{ ref('stg_oil') }}
),

holidays AS (
    SELECT * FROM {{ ref('stg_holidays') }}
),

transactions AS (
    SELECT * FROM {{ ref('stg_transactions') }}
),

-- Join everything together
joined AS (
    SELECT
        -- Identifiers
        t.date,
        t.store_nbr,
        t.family,

        -- Target variable
        t.sales,

        -- Promotion flag
        t.onpromotion,

        -- Date features
        t.year,
        t.month,
        t.day,
        t.day_of_week,
        t.week_of_year,
        t.is_weekend,

        -- Quarter
        CASE
            WHEN t.month IN (1,2,3)   THEN 1
            WHEN t.month IN (4,5,6)   THEN 2
            WHEN t.month IN (7,8,9)   THEN 3
            WHEN t.month IN (10,11,12) THEN 4
        END AS quarter,

        -- Store metadata
        s.city,
        s.state,
        s.store_type,
        s.cluster,

        -- Oil price (forward filled)
        o.oil_price,

        -- Holiday flag
        CASE WHEN h.date IS NOT NULL THEN 1 ELSE 0 END AS is_national_holiday,
        h.holiday_name,

        -- Transactions (foot traffic)
        COALESCE(tr.transactions, 0) AS transactions

    FROM train t
    LEFT JOIN stores s
        ON t.store_nbr = s.store_nbr
    LEFT JOIN oil o
        ON t.date = o.date
    LEFT JOIN holidays h
        ON t.date = h.date
    LEFT JOIN transactions tr
        ON t.date = tr.date AND t.store_nbr = tr.store_nbr
),

-- Add lag features (sales from previous periods)
with_lags AS (
    SELECT
        *,

        -- Lag features: what did this store+family sell in previous periods?
        LAG(sales, 7)  OVER (PARTITION BY store_nbr, family ORDER BY date) AS sales_lag_7,
        LAG(sales, 14) OVER (PARTITION BY store_nbr, family ORDER BY date) AS sales_lag_14,
        LAG(sales, 28) OVER (PARTITION BY store_nbr, family ORDER BY date) AS sales_lag_28,

        -- Rolling averages
        AVG(sales) OVER (
            PARTITION BY store_nbr, family
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS sales_rolling_7day_avg,

        AVG(sales) OVER (
            PARTITION BY store_nbr, family
            ORDER BY date
            ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
        ) AS sales_rolling_28day_avg

    FROM joined
)

SELECT * FROM with_lags
ORDER BY date, store_nbr, family