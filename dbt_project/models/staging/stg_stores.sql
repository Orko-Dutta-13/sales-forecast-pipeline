WITH source AS (
    SELECT * FROM {{ source('raw', 'raw_stores') }}
),

cleaned AS (
    SELECT
        CAST(store_nbr AS INTEGER)  AS store_nbr,
        UPPER(TRIM(city))           AS city,
        UPPER(TRIM(state))          AS state,
        UPPER(TRIM(type))           AS store_type,
        CAST(cluster AS INTEGER)    AS cluster
    FROM source
)

SELECT * FROM cleaned