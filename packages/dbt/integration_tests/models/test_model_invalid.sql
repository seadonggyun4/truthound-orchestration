{{
    config(
        materialized='view'
    )
}}

-- Model with invalid data for testing failing checks
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
from {{ ref('test_invalid_data') }}
