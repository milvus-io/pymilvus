# Design Documents

Design documents for PyMilvus features and architectural changes.

## Naming Convention

```
pymilvus-<NNN>-<short-name>-design.md
```

| Part | Description | Example |
|------|-------------|---------|
| `pymilvus` | Project prefix | `pymilvus` |
| `<NNN>` | Sequential number, zero-padded to 3 digits | `001`, `002` |
| `<short-name>` | Kebab-case summary | `connection-manager` |
| `-design.md` | Suffix | `-design.md` |

Examples:
- `pymilvus-001-global-client-design.md`
- `pymilvus-002-connection-manager-design.md`

Use `pymilvus-000-template.md` as a starting point for new documents.

## Required Header

Every design document must start with:

```markdown
# [Title]

- **Created:** YYYY-MM-DD
- **Updated:** YYYY-MM-DD
- **Author(s):** @github-handle
```

## Index

| Number | Title | Author | Created |
|--------|-------|--------|---------|
| 001 | [Global Client Design](pymilvus-001-global-client-design.md) | @bigsheeper | 2026-01-28 |
| 002 | [Connection Manager Design](pymilvus-002-connection-manager-design.md) | @XuanYang-cn | 2026-02-03 |
