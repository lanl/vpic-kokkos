#!/bin/bash

KEYWORD=$1
FILE_1=$2
FILE_2=$3


# Grep for the keyword
echo
echo
echo "grepping files"
TMP_1=".grepped_file_1"
TMP_2=".grepped_file_2"

grep "${KEYWORD}" $FILE_1 > $TMP_1
echo "${TMP_1} grepped."
grep "${KEYWORD}" $FILE_2 > $TMP_2
echo "${TMP_2} grepped."

# Use python to parse the files
# and combine the results
echo
echo "Using python to analyze."
time python3 parse_currents.py $TMP_1
time python3 parse_currents.py $TMP_2

echo
echo "Checking line numbers."
echo "These need to be EXACTLY the same as the total particle number! "
wc -l $TMP_1
wc -l $TMP_2
echo

# Now diff
echo
echo "Now diffing the difference."
diff -C 0 $TMP_1 $TMP_2 > "diff_currents.txt"

# Now that the test is over
# erase the temp files.
rm $TMP_1 $TMP_2
echo
echo
