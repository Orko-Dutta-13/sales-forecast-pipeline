WITH source AS (
    SELECT * FROM {{ source('raw', 'raw_oil') }}
),

-- Generate a complete date spine (no gaps)
all_dates AS (
    SELECT UNNEST(
        generate_series(
            (SELECT MIN(date) FROM source),
            (SELECT MAX(date) FROM source),
            INTERVAL '1 day'
        )
    ) AS date
),

-- Join to get nulls on missing days
oil_with_gaps AS (
    SELECT
        d.date,
        o.dcoilwtico AS oil_price
    FROM all_dates d
    LEFT JOIN source o ON d.date = o.date
),

-- Forward fill: carry last known price forward into null days
oil_filled AS (
    SELECT
        date,
        oil_price,
        COALESCE(
            -- Forward fill: carry last known price forward
            LAST_VALUE(oil_price IGNORE NULLS) OVER (
                ORDER BY date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ),
            -- Backward fill fallback: for nulls at the very start, use the first known price
            FIRST_VALUE(oil_price IGNORE NULLS) OVER (
                ORDER BY date
                ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
            )
        ) AS oil_price_filled
    FROM oil_with_gaps
)

SELECT
    CAST(date AS DATE) AS date,
    oil_price_filled   AS oil_price
FROM oil_filled
WHERE date IS NOT NULL