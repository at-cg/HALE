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
      --reads <file.fa|file.fa.gz|file.fq|file.fq.gz> \\
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
# Logging
########################################
log() {
    echo "[HALE $(date '+%Y-%m-%d %H:%M:%S')] $*"
}

########################################
# FASTA to FASTQ Conversion
########################################
WAS_CONVERTED=0
# Check if input is FASTA (case insensitive matching)
if [[ "${READS_FASTQ,,}" =~ \.(fa|fasta)(\.gz)?$ ]]; then
    # log "Input detected as FASTA. Converting to FASTQ with dummy quality 'I'..."
    
    # Generate temporary filename
    BASE_TMP=$(basename "${READS_FASTQ}")
    # Remove extensions to get clean base
    CLEAN_BASE="${BASE_TMP%.*}"
    CLEAN_BASE="${CLEAN_BASE%.*}" 
    CONVERTED_READS="${CLEAN_BASE}_converted.fastq"
    
    # 1. seqkit seq -w 0: Linearizes fasta (unwraps multiline sequences)
    # 2. awk: Converts header '>' to '@', prints sequence, prints '+', 
    #         prints 'I' string of matching length
    seqkit seq -w 0 "${READS_FASTQ}" | \
    awk '{
        if(NR%2==1){ 
            print "@"substr($0,2) 
        } else { 
            print $0; 
            print "+"; 
            gsub(/./, "I"); 
            print 
        }
    }' > "${CONVERTED_READS}"

    # Update READS_FASTQ to point to the new converted file
    READS_FASTQ="${CONVERTED_READS}"
    WAS_CONVERTED=1
    
    log "Conversion complete: Using ${READS_FASTQ} for processing"
fi

########################################
# Detect gzip input
########################################
# Reset detection based on the potentially new READS_FASTQ
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

    # Cleanup converted fastq if it was generated
    if [[ "${WAS_CONVERTED}" -eq 1 ]]; then
        log "Cleaning up converted temporary FASTQ"
        rm "${READS_FASTQ}"
    fi

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

# Cleanup converted fastq if it was generated
if [[ "${WAS_CONVERTED}" -eq 1 ]]; then
    log "Cleaning up converted temporary FASTQ"
    rm "${READS_FASTQ}"
fi

log "ONT correction completed"
log "Final output: ${HALE_OUT}"