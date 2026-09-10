import os
import csv
import json 
from utils import rm_begriffserklaerung
from utils import PROJECT_ROOT


def get_wikidir_annotation(dialect):
  fName = f"{PROJECT_ROOT}/data/wikidir/annotation_files/{dialect}.csv"
  with open(fName, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
      if r["Titel"]:
        de_term = r["Titel"]
        dial_var = r["Dialekttitel"]
        yield {"de_term": rm_begriffserklaerung(de_term), "dial_var": rm_begriffserklaerung(dial_var), "label": "yes"}

      dial_var = r["Erwähnungen"]
      label = "yes" if r["korrekt?"] == "TRUE" else "no"
      yield {"de_term": de_term, "dial_var": dial_var, "label": label}


def get_raw_dialemma_data(dev_test, dial, pos="ALL", n=-1):
  x, y = [], []
  fName = f"{PROJECT_ROOT}/data/recognition_{dev_test}.csv"
  with open(fName, "r") as f:
    reader = csv.DictReader(f)
    i = 0
    for r in reader:
      if r["pos"] == pos or pos == "ALL":
        x.append(r)
        y.append(r["label"])
        i += 1
        if i == n:
          break
  return x, y


def get_raw_dataset(dataset, dialect, split):
  fName = f"{PROJECT_ROOT}/data/{dataset}/{dataset}_{dialect}_{split}.jsonl"
  with open(fName) as f:
    for row in f:
      yield json.loads(row)
