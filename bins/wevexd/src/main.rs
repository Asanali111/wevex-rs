use clap::Parser;

/// Wevex daemon. Started by `wevex up`; not meant to be run by hand.
#[derive(Parser)]
#[command(name = "wevexd", version = wevex_core::VERSION, about)]
struct Cli {}

fn main() {
    let _cli = Cli::parse();
}
