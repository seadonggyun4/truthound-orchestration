{{
    config(
        materialized='view'
    )
}}

-- Model for testing referential integrity
select
    order_id,
    product_id,
    customer_id,
    quantity,
    total_amount,
    cast(order_date as date) as order_date,
    status
from {{ ref('test_orders') }}
