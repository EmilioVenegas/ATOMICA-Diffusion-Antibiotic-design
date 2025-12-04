#!/bin/bash
# Script to generate MSA for 3PBQ using HH-suite

set -e

FASTA_FILE="inputs/3PBQ_sequence.fasta"
OUTPUT_DIR="inputs/msa"
OUTPUT_A3M="${OUTPUT_DIR}/3PBQ.a3m"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Generating MSA for 3PBQ..."
echo "Input FASTA: ${FASTA_FILE}"
echo "Output MSA: ${OUTPUT_A3M}"

# Check if HH-suite is installed
# Try direct command first, then conda
if command -v hhblits &> /dev/null; then
    HHBLITS_CMD="hhblits"
elif conda run -n base hhblits -h &> /dev/null; then
    HHBLITS_CMD="conda run -n base hhblits"
    echo "Using HH-suite from conda base environment"
else
    echo "ERROR: HH-suite (hhblits) is not installed."
    echo ""
    echo "Installation options:"
    echo "1. Using conda/micromamba:"
    echo "   conda install -c bioconda hhsuite"
    echo ""
    echo "2. Using Homebrew (macOS):"
    echo "   brew install hhsuite"
    echo ""
    echo "3. Download from: https://github.com/soedinglab/hh-suite"
    echo ""
    echo "You also need a database. Common options:"
    echo "- UniClust30: https://uniclust.mmseqs.com/"
    echo "- UniRef30: https://www.uniprot.org/help/uniref"
    echo ""
    exit 1
fi

# Check if database is set
if [ -z "$HHSUITE_DB" ]; then
    echo "WARNING: HHSUITE_DB environment variable not set."
    echo "Please set it to your database path, e.g.:"
    echo "  export HHSUITE_DB=/path/to/uniclust30"
    echo ""
    echo "Or specify database path directly:"
    echo "  hhblits -i ${FASTA_FILE} -d /path/to/uniclust30 -o ${OUTPUT_A3M}"
    exit 1
fi

# Generate MSA
echo "Running hhblits..."
${HHBLITS_CMD} \
    -i "${FASTA_FILE}" \
    -d "${HHSUITE_DB}" \
    -o "${OUTPUT_A3M}" \
    -cpu 4 \
    -maxfilt 50000 \
    -neffmax 20 \
    -cov 25 \
    -id 99

if [ -f "${OUTPUT_A3M}" ]; then
    echo "✓ MSA generated successfully: ${OUTPUT_A3M}"
    echo "Update your YAML template to use: msa: \"${OUTPUT_A3M}\""
else
    echo "ERROR: MSA generation failed"
    exit 1
fi

