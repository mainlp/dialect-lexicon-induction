echo "Setup DiaLemma experiments"

# source: https://stackoverflow.com/a/246128 (CC-BY-SA 4.0)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$SCRIPT_DIR/../data/dialemma
SRC_DIR=$SCRIPT_DIR/../src

# Download dialemma data
wget -P $DATA_DIR https://raw.githubusercontent.com/mainlp/dialemma/refs/heads/main/data/recognition_dev.csv
wget -P $DATA_DIR https://raw.githubusercontent.com/mainlp/dialemma/refs/heads/main/data/recognition_test.csv

# concatenate both files, excluding the header in the second file
cat $DATA_DIR/recognition_dev.csv <(tail -n +2 data/dialemma/recognition_test.csv) > data/dialemma/dialemma_bar.csv
rm $DATA_DIR/recognition_dev.csv
rm $DATA_DIR/recognition_test.csv

# Download mistral data
wget -P $DATA_DIR https://raw.githubusercontent.com/mainlp/dialemma/refs/heads/main/results/test/recognition/mistral-large.csv

# create splits
python $SRC_DIR/create_splits.py --train_split 0.8 --dataset dialemma
