# wevex

Local context bus for AI coding agents. One CLI, one background daemon, the same code on macOS and Windows.

This is the Rust rewrite that replaces the Python ([wevex](https://github.com/Asanali111/wevex), v0.2.2) and Swift versions. **Work in progress, not ready for use yet.**

## Targets

| Platform | Target | Status |
|---|---|---|
| macOS, Apple Silicon | `aarch64-apple-darwin` | primary |
| macOS, Intel | `x86_64-apple-darwin` | shipped in a universal binary |
| Windows | `x86_64-pc-windows-msvc` | supported |

## Layout

```
crates/core      storage, migrations, scopes, entities, hybrid recall
crates/embed     local embeddings (bge-small via candle)
crates/mcp       MCP server, 127.0.0.1:8765/mcp
crates/watchers  file, git and transcript watchers
crates/service   launchd (macOS) / Scheduled Task (Windows)
bins/wevex       the CLI
bins/wevexd      the daemon, started by `wevex up`
```

## Build

```
cargo build --release
```

## License

Apache-2.0
