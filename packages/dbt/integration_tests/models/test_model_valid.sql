{{
    config(
        materialized='view'
    )
}}

-- Model with valid data for testing passing checks
select
    id,
    email,
    age,
    status,
    amount,
    cast(created_at as date) as created_at,
    user_uuid,
    phone,
    url
from {{ ref('test_valid_data') }}
