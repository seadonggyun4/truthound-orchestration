{% macro truthound_depot_release_tag(depot_id, asset_id, release_tag, metadata={}) %}
  {{ return(tojson({
    "operation_type": "release_tag",
    "depot_id": depot_id,
    "asset_id": asset_id,
    "release_tag": release_tag,
    "metadata": metadata
  })) }}
{% endmacro %}
