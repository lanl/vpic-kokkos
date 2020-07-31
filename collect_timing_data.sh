#!/bin/bash

# ****************************************************
# Command line arguments. 1: Deck, 2: CPU vs GPU
DECK=$1
TYPE=$2
# ****************************************************

KEYWORD="advance_p"
TEST="TEST"
DEVEL="devel"
PARSER="parse_advance_p.py"
GATHERER="gather_timings.py"

TDIR="/home/meese_wj/VPIC/vpic-kokkos/build_${DECK}_${TEST}"
DDIR="/home/meese_wj/VPIC/stable_vpic/vpic-kokkos/build_${DECK}_${DEVEL}"

# Build directories for the stderr
TEST_STDERR="${TDIR}/output.${DECK}_${TEST}"
DEVEL_STDERR="${DDIR}/output.${DECK}_${DEVEL}"

# Filenames for individual timing output
TEST_TIMINGS="${TDIR}/advance_p_${DECK}_${TEST}.txt"
DEVEL_TIMINGS="${DDIR}/advance_p_${DECK}_${DEVEL}.txt"

# Timing Table locations
TIMING_DIR="/home/meese_wj/VPIC/zigzag_timings_${DECK}"
if [ ! -d "${TIMING_DIR}" ]
then
    mkdir ${TIMING_DIR}
fi
TIMING_TABLE="${TIMING_DIR}/advance_p_timings_${DECK}_${TYPE}.txt"

# Grep individual advance_p data. This overwrites the output files.
grep $KEYWORD $TEST_STDERR > $TEST_TIMINGS
grep $KEYWORD $DEVEL_STDERR > $DEVEL_TIMINGS

# Replace output files with stripped data (Per column)
python3 $PARSER --advance_p_file=$TEST_TIMINGS --output_file=$TEST_TIMINGS \
                --deck=$DECK --version=$TEST
python3 $PARSER --advance_p_file=$DEVEL_TIMINGS --output_file=$DEVEL_TIMINGS \
                --deck=$DECK --version=$DEVEL

# Gather timings and combine data files
python3 $GATHERER --test_file=$TEST_TIMINGS --devel_file=$DEVEL_TIMINGS \
                  --output_file=$TIMING_TABLE --deck=$DECK


