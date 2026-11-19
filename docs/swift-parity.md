# Swift bridge parity

The Swift build (`~/Documents/wevex-swift`) runs on the Mac until this Rust
core can replace it. Both read the same database. This file lists the Swift
behaviour Rust has to match, and is updated whenever the Swift bridge changes.

## Scopes

| Rule | Swift source | Rust |
|---|---|---|
| A session's project is the nearest ancestor of its working directory holding `.git`, below the home folder: handle `project:<slug of folder name>` | `Hooks.swift` `ScopeResolver.projectHandle` | todo |
| The home folder is never a project, even when it is a git repo. Anything outside a repo resolves to `project:general` (added 2026-11-19; the live DB's old `project:ameliomar` scope was renamed to it) | same | todo |
| Slug: lowercase, characters outside `a-z0-9._-` become `-`, trimmed; empty becomes `default` | `ScopeResolver.slug` | todo |
| Unknown `project:` scopes are created on demand under `scope:personal` | `Storage.ensureProjectScope` | todo |
| Recall with no scope searches every scope | `RecallEngine` | todo |

## Hooks (`wevex hook <event>`, Claude Code)

| Event | Behaviour | Rust |
|---|---|---|
| `session-start` | Briefing for the project: last sessions (`conversation` fragments, 30 days, up to 4) and recent non-conversation fragments (up to 6) | todo |
| `user-prompt-submit` | Appends `U: <prompt>` (≤240 chars) to the session digest `session:<id>`; skips `/` commands and `<`-prefixed harness text; injects up to 3 relevant memories once per session | todo |
| `stop` | Appends `A: <closing message>` (≤500 chars), replacing a previous `A:` line from the same turn | todo |
| `post-tool-use` | A `git commit` in a repo becomes a `state` fragment: `Commit <short> in <project>: <subject>` | todo |
| Digest | One `conversation` fragment per session, header `Session <date> in <project>`, last 20 lines kept | todo |
| Errors | Never fail the hook; log to `~/.config/wevex/logs/hooks.log` | todo |
| Testing | `WEVEX_DB` overrides the database path | todo |

## MCP tools

| Tool | Rust |
|---|---|
| `recall(query, scope?)` | todo |
| `remember(content, scope, type?)` | todo |
| `project_briefing(scope)` | todo |

## CLI

`up`, `down`, `restart`, `status`, `doctor`, `note`, `recall`, `briefing`
(`connect` is a stub in Swift too).

## Planned, not in Swift yet

- `/wlog`: the agent writes a conversation summary and passes it to a
  `log_conversation` MCP tool, which saves it in the Obsidian vault (found via
  Obsidian's `obsidian.json`) and indexes it. Design: vault note
  `20 Projects/Wevex/01_Vision.md`.
