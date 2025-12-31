{{
    config(
        materialized='view'
    )
}}

-- Reference model for FK testing
select
    id,
    name,
    category
from {{ ref('test_reference_table') }}
