echo "Setup WikiDIR experiments"

# source: https://stackoverflow.com/a/246128 (CC-BY-SA 4.0)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DATA_FOLDER=$SCRIPT_DIR/../data/wikidir/annotation_files
SRC_DIR=$SCRIPT_DIR/../src
URL=https://huggingface.co/datasets/rlitschk/wikidir/resolve/main/annotation_files

dialects="als ksh nds pfl bar"

echo "Downloading WikiDIR files"
for dialect in $dialects
do
  # Download only if file does not exist
  if [ ! -f $DATA_FOLDER/$dialect.csv ]; then
    wget -P $DATA_FOLDER $URL/$dialect.csv
  fi
done

echo "Creating WikiDIR data splits"
python $SRC_DIR/create_splits.py --train_split 0.8 --dataset wikidir
