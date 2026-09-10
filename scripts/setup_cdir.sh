#!/bin/bash

echo "Setup CDIR experiments"

# source: https://stackoverflow.com/a/246128 (CC-BY-SA 4.0)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SRC_DIR=$SCRIPT_DIR/../src
TGT_DIR=$SCRIPT_DIR/../data/bm25

mkdir -p $TGT_DIR

# Download files
python $SRC_DIR/download_cdir.py $TGT_DIR

# List of languages/dialects to process
LANG_VALUES=("als" "pfl" "ksh" "bar" "nds")

# Loop through each LANG value
for LANG in "${LANG_VALUES[@]}"; do
    echo "Processing language: $LANG"

    python -m pyserini.index \
      --input $TGT_DIR/de.$LANG/index/ \
      --collection JsonCollection \
      --generator DefaultLuceneDocumentGenerator \
      --index $TGT_DIR/de.$LANG/index/ \
      --threads 10 \
      --language $LANG \
      --storePositions --storeDocvectors --storeRaw \
      --stemmer none \
      --storeContents \
      --keepStopwords

    echo "Completed processing language: $LANG"
done

# https://github.com/castorini/anserini/blob/master/src/main/java/io/anserini/index/IndexCollection.java
