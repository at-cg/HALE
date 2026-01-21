// clap: command line argument parser
use clap::{Args, Parser, Subcommand};

use hale::{error_correction, AlnMode};

// high-performance memory allocator
use jemallocator::Jemalloc;
// use dhat::{Alloc, Profiler};

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
// #[global_allocator]
// static ALLOC: Alloc = Alloc;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Subcommand used for error-correcting reads")]
    Correct(CorrectArgs),
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
struct CorrectArgs {
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
        short = 't',
        default_value = "128",
        help = "number of threads"
    )]
    n_threads: usize,

    #[arg(
        short = 'd',
        default_value = "30",
        help = "depth/coverage of the dataset"
    )]
    coverage: usize,

    #[arg(
        short = 'p',
        default_value = "2",
        help = "ploidy of the genome"
    )]
    ploidy: usize,

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
    // let _profiler = Profiler::new_heap();

    let cli = Cli::parse();

    match cli.command {
        Commands::Correct(args) => {
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
                args.n_threads,
                args.coverage,
                args.ploidy,
                mode,
                &args.module,
            );
        }
    }
}
