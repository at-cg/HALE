#!/usr/bin/env bash
set -euo pipefail

########################################
# Determine absolute path of the wrapper itself
########################################
WRAPPER_PATH="$(readlink -f "$0")"            # absolute path to wrapper
BIN_DIR="$(dirname "$WRAPPER_PATH")"          # directory containing wrapper
HALE_REAL="${BIN_DIR}/HALE"                   # binary next to wrapper
PIPELINE_SCRIPT="${BIN_DIR}/scripts/hale_correct_pipeline.sh"

########################################
# Check that HALE binary exists
########################################
if [[ ! -x "$HALE_REAL" ]]; then
    echo "Error: HALE binary not found at $HALE_REAL"
    exit 1
fi

# Check that pipeline exists
if [[ ! -x "$PIPELINE_SCRIPT" ]]; then
    echo "Error: Pipeline script not found at $PIPELINE_SCRIPT"
    exit 1
fi

########################################
# Wrapper help
########################################
wrapper_help() {
    cat << EOF
HALE — Haplotype-Aware Long-read Error Correction

Usage:
  hale correct [OPTIONS]

Required options:
  --reads <file>        Input FASTQ or FASTQ.GZ
  --threads <int>       Number of threads
  --depth <int>         Sequencing depth
  --ploidy <int>        Genome ploidy
  --tech <ont|hifi>     Sequencing technology

Example:
  $0 correct \\
      --reads /path/to/sample.fastq.gz \\
      --threads 32 \\
      --depth 60 \\
      --ploidy 2 \\
      --tech ont
EOF
}

########################################
# Dispatch
########################################
if [[ $# -eq 0 ]]; then
    wrapper_help
    exit 0
fi

case "$1" in
    correct)
        shift
        exec "$PIPELINE_SCRIPT" "$HALE_REAL" "$@"
        ;;
    -h|--help|help)
        wrapper_help
        ;;
    *)
        echo "Unknown command: $1"
        echo
        wrapper_help
        exit 1
        ;;
esac
