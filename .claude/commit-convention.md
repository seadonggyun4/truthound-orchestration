# Commit Convention

## Commit Message Format

```
<type>(<scope>): <description>
```

### Type

| Type | Description |
|------|-------------|
| feat | New feature |
| fix | Bug fix |
| docs | Documentation changes |
| style | Code style changes (formatting, whitespace) |
| refactor | Code refactoring without feature changes |
| test | Adding or modifying tests |
| chore | Build, config, or maintenance tasks |
| perf | Performance improvements |

### Scope (Optional)

| Scope | Description |
|-------|-------------|
| common | Common module |
| airflow | Airflow integration |
| dagster | Dagster integration |
| prefect | Prefect integration |
| dbt | dbt integration |

## Rules

1. **One-line message only** - No multi-line commit messages
2. **Year must be 2025** - Always use current year for commit dates
3. **No sensitive file names** - Do not expose internal file names (e.g., CLAUDE.md) in commit messages
4. **No Claude Code contributor** - Do not include AI assistant attribution in commits
5. **Lowercase description** - Start description with lowercase letter
6. **No period at end** - Do not end description with a period

## Examples

```
feat(common): add validation result builder
fix(airflow): resolve operator initialization error
docs: update installation guide
test(common): add unit tests for config module
chore: update dependencies
```

## Bad Examples

```
feat(common): Add validation result builder.    # Uppercase, period
docs: update CLAUDE.md                          # Exposes sensitive file
chore: add CLAUDE.md to gitignore               # Exposes sensitive file
```
