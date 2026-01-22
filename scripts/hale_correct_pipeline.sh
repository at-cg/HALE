#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# First argument: path to HALE binary
# -------------------------
HALE_BIN="$1"
shift

# -------------------------
# Parse user options
# -------------------------
THREADS=48
SEQ_TECH=""
READS_FASTQ=""
DEPTH=""
PLOIDY=""

usage() {
    echo "Usage: $0 <hale_binary> --reads <file> --threads <int> --depth <int> --ploidy <int> --tech <ont|ont_simplex|simplex|hifi|pacbio>"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reads)   READS_FASTQ="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --depth)   DEPTH="$2"; shift 2 ;;
        --ploidy)  PLOIDY="$2"; shift 2 ;;
        --tech)    SEQ_TECH="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# -------------------------
# Validate inputs
# -------------------------
: "${READS_FASTQ:?Missing --reads}"
: "${THREADS:?Missing --threads}"
: "${DEPTH:?Missing --depth}"
: "${PLOIDY:?Missing --ploidy}"
: "${SEQ_TECH:?Missing --tech}"

# -------------------------
# Normalize technology
# -------------------------
TECH_LOWER="$(echo "${SEQ_TECH}" | tr '[:upper:]' '[:lower:]')"
case "${TECH_LOWER}" in
    ont|ont_simplex|simplex)
        MODE="ont"
        ALIGN_TECH="ont_simplex"
        ;;
    hifi|pacbio)
        MODE="hifi"
        ALIGN_TECH="hifi"
        ;;
    *)
        echo "Error: unsupported technology '${SEQ_TECH}'"
        exit 1
        ;;
esac

# -------------------------
# Resolve paths
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALIGN_SCRIPT="${SCRIPT_DIR}/create_batched_alignments.sh"

if [[ ! -x "${HALE_BIN}" ]]; then
    echo "Error: HALE binary not found at ${HALE_BIN}"
    exit 1
fi

if [[ ! -x "${ALIGN_SCRIPT}" ]]; then
    echo "Error: create_batched_alignments.sh not found or not executable at ${ALIGN_SCRIPT}"
    exit 1
fi

# -------------------------
# Input name handling
# -------------------------
READS_BASENAME="$(basename "${READS_FASTQ}")"
INPUT_NAME="${READS_BASENAME%.fastq}"
INPUT_NAME="${INPUT_NAME%.fq}"
INPUT_NAME="${INPUT_NAME%.fastq.gz}"
INPUT_NAME="${INPUT_NAME%.fq.gz}"

# -------------------------
# Logging function
# -------------------------
log() {
    echo "[HALE $(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Input reads : ${READS_FASTQ}"
log "Input name  : ${INPUT_NAME}"
log "Threads     : ${THREADS}"
log "Depth       : ${DEPTH}"
log "Ploidy      : ${PLOIDY}"
log "Technology  : ${MODE}"

# -------------------------
# HiFi pipeline
# -------------------------
if [[ "${MODE}" == "hifi" ]]; then
    log "Running HiFi correction pipeline"

    seqkit seq -ni "${READS_FASTQ}" > "${INPUT_NAME}.rids"

    log "Starting minimap2 all-vs-all overlap"
    /usr/bin/time "$ALIGN_SCRIPT" \
        "${READS_FASTQ}" \
        "${INPUT_NAME}.rids" \
        "${THREADS}" \
        "hifi_batch_alignments_1_${INPUT_NAME}" \
        "${ALIGN_TECH}" \
        1

    log "Running HALE correction"
    /usr/bin/time "$HALE_BIN" correct \
        --read-alns "hifi_batch_alignments_1_${INPUT_NAME}" \
        -m hale \
        -t "${THREADS}" \
        -d "${DEPTH}" \
        -p "${PLOIDY}" \
        "${READS_FASTQ}" \
        "hale_corrected_${READS_BASENAME}"

    log "Cleaning up temporary files"
    rm -r "hifi_batch_alignments_1_${INPUT_NAME}" "${INPUT_NAME}.rids"

    log "HiFi correction completed"
    log "Final output: hale_corrected_${READS_BASENAME}"
    exit 0
fi

# -------------------------
# ONT pipeline
# -------------------------
log "Running ONT correction pipeline: minimap2 -> HALE pre-correction -> minimap2 -> HALE pre-correction -> minimap2 -> HALE final correction"

# Round 1
seqkit seq -ni "${READS_FASTQ}" > "${INPUT_NAME}.rids"
log "Starting minimap2 all-vs-all overlap round 1"
/usr/bin/time "$ALIGN_SCRIPT" \
    "${READS_FASTQ}" \
    "${INPUT_NAME}.rids" \
    "${THREADS}" \
    "ont_batch_alignments_1_${INPUT_NAME}" \
    "${ALIGN_TECH}" \
    1

log "Running HALE pre-correction round 1"
/usr/bin/time "$HALE_BIN" correct \
    --read-alns "ont_batch_alignments_1_${INPUT_NAME}" \
    -m pih \
    -t "${THREADS}" \
    -d "${DEPTH}" \
    -p "${PLOIDY}" \
    "${READS_FASTQ}" \
    "pih_corrected_${READS_BASENAME}"

# Round 2
seqkit seq -ni "pih_corrected_${READS_BASENAME}" > "pih_corrected_${INPUT_NAME}.rids"
log "Starting minimap2 all-vs-all overlap round 2"
/usr/bin/time "$ALIGN_SCRIPT" \
    "pih_corrected_${READS_BASENAME}" \
    "pih_corrected_${INPUT_NAME}.rids" \
    "${THREADS}" \
    "ont_batch_alignments_2_${INPUT_NAME}" \
    "${ALIGN_TECH}" \
    2

log "Running HALE pre-correction round 2"
/usr/bin/time "$HALE_BIN" correct \
    --read-alns "ont_batch_alignments_2_${INPUT_NAME}" \
    -m pih \
    -t "${THREADS}" \
    -d "${DEPTH}" \
    -p "${PLOIDY}" \
    "pih_corrected_${READS_BASENAME}" \
    "pih_corrected_2_${READS_BASENAME}"

# Round 3
seqkit seq -ni "pih_corrected_2_${READS_BASENAME}" > "pih_corrected_2_${INPUT_NAME}.rids"
log "Starting minimap2 all-vs-all overlap round 3"
/usr/bin/time "$ALIGN_SCRIPT" \
    "pih_corrected_2_${READS_BASENAME}" \
    "pih_corrected_2_${INPUT_NAME}.rids" \
    "${THREADS}" \
    "ont_batch_alignments_3_${INPUT_NAME}" \
    "${ALIGN_TECH}" \
    3

log "Running HALE final correction round 3"
/usr/bin/time "$HALE_BIN" correct \
    --read-alns "ont_batch_alignments_3_${INPUT_NAME}" \
    -m hale \
    -t "${THREADS}" \
    -d "${DEPTH}" \
    -p "${PLOIDY}" \
    "pih_corrected_2_${READS_BASENAME}" \
    "hale_corrected_${READS_BASENAME}"

# Cleanup
log "Deleting temporary files"
rm \
    "pih_corrected_${READS_BASENAME}" \
    "pih_corrected_${INPUT_NAME}.rids" \
    "pih_corrected_2_${READS_BASENAME}" \
    "pih_corrected_2_${INPUT_NAME}.rids"

rm -r \
    "ont_batch_alignments_1_${INPUT_NAME}" \
    "ont_batch_alignments_2_${INPUT_NAME}" \
    "ont_batch_alignments_3_${INPUT_NAME}"

log "ONT correction completed"
log "Final output: hale_corrected_${READS_BASENAME}"
