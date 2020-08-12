#!/bin/bash

ZIGZAG_FILE=$1
DEVEL_FILE=$2
KEYWORD=$3

OUTPUT="diff_${KEYWORD}.txt"
# Grep and diff for the keyword
echo
echo "Grepping files for $KEYWORD and diffing the result..."
time diff -C 0 <(grep "$KEYWORD" $ZIGZAG_FILE) <(grep "$KEYWORD" $DEVEL_FILE) > "$OUTPUT"

echo
echo "diffed file is of size"
wc -l $OUTPUT
echo

echo "Now showing the file."
less $OUTPUT

