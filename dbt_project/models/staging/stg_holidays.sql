WITH source AS (
    SELECT * FROM {{ source('raw', 'raw_holidays') }}
),

cleaned AS (
    SELECT
        CAST(date AS DATE)          AS date,
        UPPER(TRIM(type))           AS holiday_type,
        UPPER(TRIM(locale))         AS locale,
        UPPER(TRIM(locale_name))    AS locale_name,
        UPPER(TRIM(description))    AS description,
        CAST(transferred AS BOOLEAN) AS is_transferred
    FROM source
),

-- Keep only true holidays (exclude transferred-away days)
-- A "transferred" holiday means the day was moved, so it's actually a work day
national_holidays AS (
    SELECT
        date,
        description AS holiday_name,
        1 AS is_national_holiday
    FROM cleaned
    WHERE locale = 'NATIONAL'
      AND holiday_type = 'HOLIDAY'
      AND is_transferred = FALSE
)

SELECT * FROM national_holidays