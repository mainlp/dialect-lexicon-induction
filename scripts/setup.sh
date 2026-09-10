#!/bin/bash

# source: https://stackoverflow.com/a/246128 (CC-BY-SA 4.0)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

$SCRIPT_DIR/setup\_dialemma.sh
$SCRIPT_DIR/setup\_wikidir.sh
$SCRIPT_DIR/setup\_cdir.sh
