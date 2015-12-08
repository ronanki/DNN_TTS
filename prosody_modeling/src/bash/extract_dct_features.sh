#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage:bash extract_dct_features.sh <Config-file>"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "Config file doesn't exist"
    exit 1
else
    source $1
fi

echo "===========================================================";
echo "Experimental condition";
echo "Date: $(date)"
echo "ROOTPATH: $ROOTDIR"
echo "===========================================================";

## Check local files
required_local_files="$CONFIG_GLOBAL"
for file in ${required_local_files}
do
    if [ ! -f $file ]; then
        echo "$file not found"
        exit 1
    else
    	source $CONFIG_GLOBAL    
    fi
done

echo "Confirming existence of local files"
echo

echo "Extracting DCT features..."
python $PY/run_dct.py \
$INTER_MODULE_F0 \
$INTER_MODULE_LAB \
$INTER_MODULE_DCT \
word \
9 
echo "done!"
echo "Date: $(date)"
