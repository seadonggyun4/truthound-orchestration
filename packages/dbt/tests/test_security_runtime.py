"""Exercise the patched SQL parser on dbt's real ephemeral-model compilation path."""

import subprocess
import sys


def test_native_cte_injection_with_patched_sqlparse(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sqlparse
from packaging.version import Version
from dbt.compilation import inject_ctes_into_sql
from dbt.contracts.graph.nodes import InjectedCTE
assert Version(sqlparse.__version__) >= Version('0.6.0')
# dbt's compiler prefixes generated CTE fragments with whitespace.
cte = InjectedCTE(id='quality_source', sql=' __dbt__cte__quality_source as (select 1 as id)')
for original in (
    'select * from __dbt__cte__quality_source',
    'with local_data as (select * from __dbt__cte__quality_source) select * from local_data',
):
    compiled = inject_ctes_into_sql(original, [cte])
    statements = sqlparse.parse(compiled)
    assert len(statements) == 1
    assert statements[0].get_type() == 'SELECT'
    assert compiled.count('__dbt__cte__quality_source as (select 1 as id)') == 1
    assert 'select * from' in compiled
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
