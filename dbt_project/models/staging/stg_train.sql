WITH source AS (
    SELECT * FROM {{ source('raw', 'raw_train') }}
),

cleaned AS (
    SELECT
        id,
        CAST(date AS DATE)          AS date,
        CAST(store_nbr AS INTEGER)  AS store_nbr,
        UPPER(TRIM(family))         AS family,
        CAST(sales AS DOUBLE)       AS sales,
        CAST(onpromotion AS INTEGER) AS onpromotion,

        -- Date parts (useful features for modeling)
        EXTRACT(YEAR FROM date)         AS year,
        EXTRACT(MONTH FROM date)        AS month,
        EXTRACT(DAY FROM date)          AS day,
        DAYOFWEEK(date)                 AS day_of_week,
        WEEKOFYEAR(date)                AS week_of_year,

        -- Flags
        CASE WHEN DAYOFWEEK(date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend

    FROM source
    WHERE sales >= 0  -- remove any negative sales (data errors)
)

SELECT * FROM cleaned