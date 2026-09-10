import os
import csv
import json 
from argparse import ArgumentParser
from dataloader import get_wikidir_annotation
from utils import rm_begriffserklaerung
from utils import PROJECT_ROOT


parser = ArgumentParser()
parser.add_argument("--train_split", type=float, help="Split ratio for the training dataset.", required=False, default=0.8)
parser.add_argument("--dataset", type=str, choices=["wikidir", "dialemma"], help="dataset", required=True)
args = parser.parse_args()


def create_wikidir_splits(dialects, train_split = 0.8):
  wikidir_directory = f"{PROJECT_ROOT}/data/wikidir/"
  for dial in dialects:
    # create train and test split for wikidir
    records = []
    for r in get_wikidir_annotation(dial):
      r["de_term"] = rm_begriffserklaerung(r["de_term"])
      records.append(r)
    n_train = int(train_split * len(records))
    n_test = len(records) - n_train
    for train_test, data in [("train", records[:n_train]), ("test", records[-n_test:])]:
      wikidir_filename = f"wikidir_{dial}_{train_test}.jsonl"
      with open(wikidir_directory + wikidir_filename, "w") as f:
        for r in data:
          f.write(json.dumps(r, ensure_ascii=False) + "\n")


def create_dialemma_split(train_split = 0.8):
  records = []
  dialemma_directory = f"{PROJECT_ROOT}/data/dialemma/"
  with open(dialemma_directory + "dialemma_bar.csv", "r") as f:
    reader = csv.DictReader(f, fieldnames=["id", "lemma_id", "de_term", "de_freq", "pos", "pos_perc", "bar",
                                           "bar_freq", "ld", "label", "contexts"])
    next(reader) # skip header
    for r in reader:
      records.append({"id": r["id"], "de_term": r["de_term"], "dial_var": r["bar"], "label": r["label"]})

  n_train = int(len(records) * train_split)
  n_test = len(records) - n_train
  for train_test, data in [("train", records[:n_train]), ("test", records[-n_test:])]:
    dialemma_filename = f"dialemma_bar_{train_test}.jsonl"
    with open(dialemma_directory + dialemma_filename, "w") as f:
      for r in data:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
  train_split = args.train_split
  dialects = ["ksh", "nds", "als", "pfl", "bar"]
  
  if args.dataset == "wikidir":
    create_wikidir_splits(dialects, train_split=train_split)
  else:
    create_dialemma_split(train_split=train_split)


if __name__ == '__main__':
    main()
