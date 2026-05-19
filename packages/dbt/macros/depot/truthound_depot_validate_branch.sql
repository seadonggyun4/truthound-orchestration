{% macro truthound_depot_validate_branch(depot_id, asset_id, branch_id, metadata={}) %}
  {{ return(tojson({
    "operation_type": "validate_branch",
    "depot_id": depot_id,
    "asset_id": asset_id,
    "branch_id": branch_id,
    "metadata": metadata
  })) }}
{% endmacro %}
