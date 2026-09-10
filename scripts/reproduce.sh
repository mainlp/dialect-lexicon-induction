# source: https://stackoverflow.com/a/246128 (CC-BY-SA 4.0)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SRC_DIR=$SCRIPT_DIR/../src


echo "Reproducing DiaLemma results"
$SCRIPT_DIR/setup_dialemma.sh
$SCRIPT_DIR/run.sh dialemma main
$SCRIPT_DIR/run.sh dialemma ablation
python $SRC_DIR/plot.py


echo "Reproducing WikiDIR results"
$SCRIPT_DIR/setup_wikidir.sh
$SCRIPT_DIR/run.sh wikidir main


if [[ -z "${JAVA_HOME}" ]]; then
  echo "Skipping CDIR experiments, because environment variable JAVA_HOME is not set."
else
  echo "Reproducing CDIR results"
  $SCRIPT_DIR/setup_cdir.sh
  python $SRC_DIR/run_cdir.py
fi
