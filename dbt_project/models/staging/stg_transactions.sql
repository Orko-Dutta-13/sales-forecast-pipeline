WITH source AS (
    SELECT * FROM {{ source('raw', 'raw_transactions') }}
),

cleaned AS (
    SELECT
        CAST(date AS DATE)              AS date,
        CAST(store_nbr AS INTEGER)      AS store_nbr,
        CAST(transactions AS INTEGER)   AS transactions
    FROM source
)

SELECT * FROM cleaned