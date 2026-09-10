# Resource-Lean Lexicon Induction for German Dialects

This repository contains the code, data and evaluation scripts to reproduce the results of the paper [Resource-Lean Lexicon Induction for German Dialects](https://lrec.elra.info/lrec2026-main-711), which has been presented at _LREC 2026_. 

## Overview

1. [Dialect Variation Dictionaries](https://github.com/mainlp/dialect-lexicon-induction#-dialect-variation-dictionaries)
2. [Reproduce Results](https://github.com/mainlp/dialect-lexicon-induction#reproduce-results)
   1. [DiaLemma BLI](https://github.com/mainlp/dialect-lexicon-induction#-dialemma-bli) (Table 2, Figure 1)
   2. [WikiDIR BLI](https://github.com/mainlp/dialect-lexicon-induction#-wikidir-bli) (Tables 3-5)
   3. [Cross-Dialect IR](https://github.com/mainlp/dialect-lexicon-induction#-cross-dialect-ir) (Table 6)
3. [Citation](https://github.com/mainlp/dialect-lexicon-induction#citation)

## 📖 Dialect Variation Dictionaries

We provide automatically generated dictionaries for five German dialects, which we evaluated extrinsically on the task of cross-dialect retrieval (query expansion). The files can be downloaded from the folder [data/multilemma/](data/multilemma).

| Dialect | Lemmas  | Variants | V/L   | File                                                          |
|---------|---------|----------|-------|---------------------------------------------------------------|
| als     | 38,129  | 88,114   | 2.31  | [als_dictionary.jsonl](data/multilemma/als_dictionary.jsonl)  |
| bar     | 27,974  | 51,392   | 1.86  | [bar_dictionary.jsonl](data/multilemma/bar_dictionary.jsonl)  |
| ksh     | 6,889   | 9,384    | 1.36  | [ksh_dictionary.jsonl](data/multilemma/ksh_dictionary.jsonl)  |
| pfl     | 9,127   | 13,050   | 1.43  | [pfl_dictionary.jsonl](data/multilemma/pfl_dictionary.jsonl)  |
| nds     | 21,974  | 39,547   | 1.80  | [nds_dictionary.jsonl](data/multilemma/nds_dictionary.jsonl)  |

The dictionaries above have been **automatically induced** following the [DiaLemma annotation framework ](https://aclanthology.org/2025.findings-emnlp.762/). We trained a classifier (annotation model) on [human-annotated Bavarian word pairs](https://github.com/mainlp/dialemma) and classified (annotated) for German reference terms their ten lexical nearest neighbors. 

**Example (als):**

```json
{
  "term": "Ortschaft",
  "variants": [
    "ortschaft",
    "Ortschàft",
    "Oortschaft"
  ],
  "inflected_variants": [
    "Ortschaftä",
    "Ortschofte",
    "Ortschafta",
    "Ortschaftè"
  ],
  "unrelated": [
    "Wûrtschaft",
    "Wértschaft",
    "Wrtschaft"
  ]
}
```

**Format:**

- `"term"`: German reference term.
- `"variants"` / `"inflected_variants"`: Dialect terms that were classified as (inflected) translations.
- `"unrelated"`: Dialect terms that were classified as being unrelated to the reference term.

# Reproduce Results

Create a python environment and install the required packages:
```
conda create --name multilemma python=3.10
conda activate multilemma
pip install -r requirements.txt
chmod +x scripts/*
```

Run `scripts/reproduce.sh` to download input files and generate result files:  
- Input files can be found in [data/](data/).
- Result files can be found in [results/](results/). 

Below are the steps to reproduce specific results. **Note:** Scripts need to be run in the project root folder. 

### 🤔 DiaLemma BLI

- Download DiaLemma files and create splits: `scripts/setup_dialemma.sh`
- Reproduce results in **Table 2**: `scripts/run.sh dialemma main`
- Reproduce results in **Figure 1**: `scripts/run.sh dialemma ablation` and `python src/plot.py`

> Results are written to [dialemma_main.csv](results/dialemma_main.csv), [dialemma_ablation.csv](results/dialemma_ablation.csv), and [Figure-1.pdf](results/Figure-1.pdf).

### 🌐 WikiDIR BLI

- Download WikiDIR files and create splits: `scripts/setup_wikidir.sh`
- Reproduce results in **Tables 3-5**: `scripts/run.sh wikidir main`

> Results are written to [wikidir_main_precision.csv](results/wikidir_main_precision.csv), [wikidir_main_recall.csv](results/wikidir_main_recall.csv), and [wikidir_main_f1.csv](results/wikidir_main_f1.csv).

### 🔍 Cross-Dialect IR

- Set the `JAVA_HOME` environment variable (e.g., `export JAVA_HOME=/path/to/java/jdk-21.0.4`).
- Download retrieval data and index dialect corpora: `scripts/setup_cdir.sh` 
- Reproduce results in **Table 6**: `python src/run_cdir.py`

> Results are written to [bm25_results.csv](results/bm25_results.csv). 

# Citation

Please consider citing our paper if you use resources from this repository:

```bibtex
@inproceedings{litschko-etal-2026-resource,
  title = {Resource-Lean Lexicon Induction for German Dialects},
  author = {Litschko, Robert and Plank, Barbara and Frassinelli, Diego},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {9044--9050},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/2feouaji2rxe},
}
```
