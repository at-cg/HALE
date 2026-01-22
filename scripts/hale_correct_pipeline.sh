#!/usr/bin/env bash
set -euo pipefail

########################################
# First argument: path to HALE binary
########################################
HALE_BIN="$1"
shift

########################################
# Defaults
########################################
THREADS=48
SEQ_TECH=""
READS_FASTQ=""
DEPTH=""
PLOIDY=""

########################################
# Usage
########################################
usage() {
    cat << EOF
Usage:
  hale correct \\
      --reads <file.fastq|file.fastq.gz> \\
      --threads <int> \\
      --depth <int> \\
      --ploidy <int> \\
      --tech <ont|ont_simplex|simplex|hifi|pacbio>
EOF
    exit 1
}

########################################
# Parse arguments
########################################
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

########################################
# Validate inputs
########################################
: "${READS_FASTQ:?Missing --reads}"
: "${THREADS:?Missing --threads}"
: "${DEPTH:?Missing --depth}"
: "${PLOIDY:?Missing --ploidy}"
: "${SEQ_TECH:?Missing --tech}"

########################################
# Detect gzip input
########################################
IS_GZIPPED=0
if [[ "${READS_FASTQ}" == *.gz ]]; then
    IS_GZIPPED=1
fi

########################################
# Normalize technology
########################################
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

########################################
# Resolve paths
########################################
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ALIGN_SCRIPT="${SCRIPT_DIR}/create_batched_alignments.sh"

if [[ ! -x "${HALE_BIN}" ]]; then
    echo "Error: HALE binary not executable: ${HALE_BIN}"
    exit 1
fi

if [[ ! -x "${ALIGN_SCRIPT}" ]]; then
    echo "Error: create_batched_alignments.sh not found or not executable"
    exit 1
fi

########################################
# Input name handling
########################################
READS_BASENAME="$(basename "${READS_FASTQ}")"

INPUT_NAME="${READS_BASENAME}"
INPUT_NAME="${INPUT_NAME%.fastq.gz}"
INPUT_NAME="${INPUT_NAME%.fq.gz}"
INPUT_NAME="${INPUT_NAME%.fastq}"
INPUT_NAME="${INPUT_NAME%.fq}"

########################################
# Logging
########################################
log() {
    echo "[HALE $(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Reads       : ${READS_FASTQ}"
log "Threads     : ${THREADS}"
log "Depth       : ${DEPTH}"
log "Ploidy      : ${PLOIDY}"
log "Technology  : ${MODE}"

########################################
# HiFi pipeline
########################################
if [[ "${MODE}" == "hifi" ]]; then
    log "Running HiFi correction pipeline"

    seqkit seq -ni "${READS_FASTQ}" > "${INPUT_NAME}.rids"

    log "Minimap2 all-vs-all overlaps"
/usr/bin/time "$ALIGN_SCRIPT" \
        "${READS_FASTQ}" \
        "${INPUT_NAME}.rids" \
        "${THREADS}" \
        "hifi_batch_alignments_1_${INPUT_NAME}" \
        "${ALIGN_TECH}" \
        1

    HALE_OUT="hale_corrected_${READS_BASENAME%.gz}"

    log "Running HALE correction"
/usr/bin/time "$HALE_BIN" correct \
        --read-alns "hifi_batch_alignments_1_${INPUT_NAME}" \
        -m hale \
        -t "${THREADS}" \
        -d "${DEPTH}" \
        -p "${PLOIDY}" \
        "${READS_FASTQ}" \
        "${HALE_OUT}"

    if [[ "${IS_GZIPPED}" -eq 1 ]]; then
        log "Compressing output FASTQ"
        gzip -f "${HALE_OUT}"
        HALE_OUT="${HALE_OUT}.gz"
    fi

    rm -r "hifi_batch_alignments_1_${INPUT_NAME}" "${INPUT_NAME}.rids"

    log "HiFi correction completed"
    log "Final output: ${HALE_OUT}"
    exit 0
fi

########################################
# ONT pipeline
########################################
log "Running ONT correction pipeline (3 rounds)"

########################################
# Round 1
########################################
seqkit seq -ni "${READS_FASTQ}" > "${INPUT_NAME}.rids"

log "Minimap2 overlap round 1"
/usr/bin/time "$ALIGN_SCRIPT" \
    "${READS_FASTQ}" \
    "${INPUT_NAME}.rids" \
    "${THREADS}" \
    "ont_batch_alignments_1_${INPUT_NAME}" \
    "${ALIGN_TECH}" \
    1

PIH1_OUT="pih_corrected_${READS_BASENAME%.gz}"

log "HALE pre-correction round 1"
/usr/bin/time "$HALE_BIN" correct \
    --read-alns "ont_batch_alignments_1_${INPUT_NAME}" \
    -m pih \
    -t "${THREADS}" \
    -d "${DEPTH}" \
    -p "${PLOIDY}" \
    "${READS_FASTQ}" \
    "${PIH1_OUT}"

########################################
# Round 2
########################################
seqkit seq -ni "${PIH1_OUT}" > "pih_corrected_${INPUT_NAME}.rids"

log "Minimap2 overlap round 2"
/usr/bin/time "$ALIGN_SCRIPT" \
    "${PIH1_OUT}" \
    "pih_corrected_${INPUT_NAME}.rids" \
    "${THREADS}" \
    "ont_batch_alignments_2_${INPUT_NAME}" \
    "${ALIGN_TECH}" \
    2

PIH2_OUT="pih_corrected_2_${READS_BASENAME%.gz}"

log "HALE pre-correction round 2"
/usr/bin/time "$HALE_BIN" correct \
    --read-alns "ont_batch_alignments_2_${INPUT_NAME}" \
    -m pih \
    -t "${THREADS}" \
    -d "${DEPTH}" \
    -p "${PLOIDY}" \
    "${PIH1_OUT}" \
    "${PIH2_OUT}"

########################################
# Round 3
########################################
seqkit seq -ni "${PIH2_OUT}" > "pih_corrected_2_${INPUT_NAME}.rids"

log "Minimap2 overlap round 3"
/usr/bin/time "$ALIGN_SCRIPT" \
    "${PIH2_OUT}" \
    "pih_corrected_2_${INPUT_NAME}.rids" \
    "${THREADS}" \
    "ont_batch_alignments_3_${INPUT_NAME}" \
    "${ALIGN_TECH}" \
    3

HALE_OUT="hale_corrected_${READS_BASENAME%.gz}"

log "HALE final correction"
/usr/bin/time "$HALE_BIN" correct \
    --read-alns "ont_batch_alignments_3_${INPUT_NAME}" \
    -m hale \
    -t "${THREADS}" \
    -d "${DEPTH}" \
    -p "${PLOIDY}" \
    "${PIH2_OUT}" \
    "${HALE_OUT}"

########################################
# Compress output if needed
########################################
if [[ "${IS_GZIPPED}" -eq 1 ]]; then
    log "Compressing final output FASTQ"
    gzip -f "${HALE_OUT}"
    HALE_OUT="${HALE_OUT}.gz"
fi

########################################
# Cleanup
########################################
rm \
    "${PIH1_OUT}" \
    "${PIH2_OUT}" \
    "pih_corrected_${INPUT_NAME}.rids" \
    "pih_corrected_2_${INPUT_NAME}.rids"

rm -r \
    "ont_batch_alignments_1_${INPUT_NAME}" \
    "ont_batch_alignments_2_${INPUT_NAME}" \
    "ont_batch_alignments_3_${INPUT_NAME}"

log "ONT correction completed"
log "Final output: ${HALE_OUT}"
