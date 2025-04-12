// clap: command line argument parser
use clap::{Args, Parser, Subcommand};

use hale::{error_correction, AlnMode};

// high-performance memory allocator
use jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Subcommand used for error-correcting reads")]
    Inference(InferenceArgs),
}

#[derive(Args)]
#[group(required = false, multiple = false)]
struct AlignmentsIO {
    #[arg(long, help = "Path to the folder containing *.oec.zst alignments")]
    read_alns: Option<String>,

    #[arg(
        long,
        help = "Path to the folder where *.oec.zst alignments will be saved"
    )]
    write_alns: Option<String>,
}


#[derive(Args)]
struct InferenceArgs {
    #[command(flatten)]
    alns: AlignmentsIO,

    #[arg(
        short = 'w',
        default_value = "4096",
        help = "Size of the window used for target chunking (default 4096)"
    )]
    window_size: u32,

    #[arg(
        short = 'b',
        default_value = "64",
        help = "Batch size per core"
    )]
    batch_size: usize,

    #[arg(
        short = 'c',
        default_value = "",
        help = "Path to a cluster of reads."
    )]
    cluster : String,

    #[arg(
        short = 'm',
        default_value = "hale",
        help = "m can be consensus, pih or hale"
    )]
    module: String,

    #[arg(help = "Path to the fastq reads (can be gzipped)")]
    reads: String,

    #[arg(help = "Path to the corrected reads")]
    output: String,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inference(args) => {
            let mode = match (args.alns.read_alns, args.alns.write_alns) {
                (None, None) => AlnMode::None,
                (Some(p), None) => AlnMode::Read(p),
                (None, Some(p)) => AlnMode::Write(p),
                _ => unreachable!(),
            };

            error_correction(
                args.reads,
                args.output,
                &args.cluster,
                args.window_size,
                args.batch_size,
                mode,
                &args.module,
            );
        }
    }
}
