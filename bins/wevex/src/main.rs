use clap::Parser;

/// Wevex: local context bus for AI coding agents.
#[derive(Parser)]
#[command(name = "wevex", version = wevex_core::VERSION, about)]
struct Cli {}

fn main() {
    let _cli = Cli::parse();
}
