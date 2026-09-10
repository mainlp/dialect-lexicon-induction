import csv
import multiprocessing
import random
import tqdm

from dataloader import get_raw_dataset
from utils import PROJECT_ROOT

random.seed(0)

from collections import defaultdict
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from featurize import *


def evaluate(y_hat, y_test, label):
  TP = sum(prediction == gold for prediction, gold in zip(y_hat, y_test) if gold == label)
  FN = sum(prediction != gold for prediction, gold in zip(y_hat, y_test) if gold == label)
  FP = sum(prediction != gold for prediction, gold in zip(y_hat, y_test) if prediction == label)
  P = (TP / (TP + FP)) if TP + FP > 0 else 0
  R = (TP / (TP + FN)) if TP + FN > 0 else 0
  denominator = (P + R)
  if denominator == 0:
    return 0, 0, 0
  F1 = 2 * P * R / denominator
  return P, R, F1


def _helper(instance, dial):
  return featurize(instance, dial), instance["label"]


dataset_dial_split2dataset = dict()
def get_featurized_dataset(dataset, dialect, split):
  if dialect in dataset_dial_split2dataset:
    return dataset_dial_split2dataset[(dataset, dialect, split)]
  else:
    x, y = [], []
    if dialect == "ALL":
      for d in ["ksh", "als", "pfl", "nds", "bar"]:
        x_d, y_d = get_featurized_dataset(dataset, d, split)
        x.extend(x_d)
        y.extend(y_d)
    else:
      with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        fn = partial(_helper, dial=dialect)
        for xi, yi in p.imap(fn, get_raw_dataset(dataset, dialect, split)):
          x.append(xi)
          y.append(yi)
    
    dataset_dial_split2dataset[(dataset, dialect, split)] = (x, y)
    return x, y


def predict_and_evaluate(model, x_test, y_test, verbose):
  if verbose: print("predicting")
  y_hat = model.predict(x_test).tolist()
  P, R, F1_yes = evaluate(y_test=y_test, y_hat=y_hat, label="yes")
  freqs = Counter(y_hat)
  if verbose: print(f"P(yes) = {P:.2f}, R(yes) = {R:.2f}, F1(yes) = {F1_yes:.2f} ({freqs['yes']} out of {sum(freqs.values())})")
  return P, R, F1_yes, freqs


def eval_x_to_x(src_dialect, tgt_dialect, src_pos="ALL", tgt_pos="ALL", src_dataset="dialemma", tgt_dataset="dialemma",
                verbose=True, ntrees=100, perc_train=1.0, perc_test=1.0, random_seed=1):
  if tgt_dialect != "bar" and tgt_dataset == "dialemma":
    if verbose: print("dialemma only possible for bar")
    return -1, -1, -1

  if verbose: print("featurizing train")
  
  # load training data
  if src_dataset == "dialemma": assert src_dialect == "bar"
  
  x_train, y_train = get_featurized_dataset(dataset=src_dataset, dialect=src_dialect, split="train")
  n_train = max(1, int(perc_train * len(x_train)))
  x_train, y_train = x_train[:n_train], y_train[:n_train]
  
  x_test, y_test = get_featurized_dataset(dataset=tgt_dataset, dialect=tgt_dialect, split="test")
  n_test = max(1, int(perc_test * len(x_test)))
  x_test, y_test = x_test[:n_test], y_test[:n_test]

  clf = RandomForestClassifier(
    n_estimators=ntrees, n_jobs=10, random_state=random_seed, verbose=verbose
  ).fit(x_train, y_train)
  
  P, R, F1_yes, freqs = predict_and_evaluate(model=clf, x_test=x_test, y_test=y_test, verbose=verbose)
  return P, R, F1_yes, n_train, n_test


def baseline_mistral():
  from utils import postprocess_recognition
  fName = f"{PROJECT_ROOT}/data/dialemma/mistral-large.csv"
  with open(fName) as f:
    reader = csv.DictReader(f, fieldnames=["id", "lemma_id", "de_term", "de_freq", "pos", "pos_perc", "bar", "bar_freq",
                                           "ld", "label", "contexts", "prediction"])
    next(reader)
    y_hat_mistral = []
    y_hat_random = []
    y = []
    for r in reader:
      if int(r["id"]) >= 80_000:
        y_hat_mistral.append(postprocess_recognition(r["prediction"]))
        y_hat_random.append(random.choice(["yes", "inflected", "no"]))
        
        y.append(r["label"])
    
  print("random")
  P, R, F1 = evaluate(y_hat=y_hat_random, y_test=y, label="yes")
  print(f"{P};{R};{F1}")
  
  print("mistral")
  P, R, F1 = evaluate(y_hat=y_hat_mistral, y_test=y, label="yes")
  print(f"{P};{R};{F1}")


def main():
  random_seed = 1
  ntrees = 100
  verbose = False


  # source dialects
  rows = ["ksh", "nds", "als", "pfl", "bar", "ALL"]
  # target dialects
  columns = ["ksh", "nds", "als", "pfl", "bar"]


  def print_results(name_results):
    for name, results in name_results.items():
      print(name)
      print(";".join(columns))
      for src in rows:
        if src in results:
          print(";".join([src] + [str(results[src][tgt]) for tgt in columns]))
      print()


  # 
  # mistral-large baseline on lrec test set
  # 
  baseline_mistral()
  # exit(0)


  #
  # dialemma -> dialemma
  #
  P, R, F1, n_train, n_test  = eval_x_to_x("bar", "bar", src_dataset="dialemma", tgt_dataset="dialemma", ntrees=ntrees, random_seed=random_seed)
  print(f"{P};{R};{F1};{n_train};{n_test}")
  # exit(0)


  #
  # ablation study (wikidir -> wikidir)
  #
  for dial in rows:
    print(dial)
    print("P;R;F1;n_train;n_test")
    for perc_train in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
      P, R, F1, n_train, n_test  = eval_x_to_x("ALL", dial, src_dataset="wikidir", tgt_dataset="wikidir", ntrees=ntrees, random_seed=random_seed, perc_train=perc_train, verbose=verbose)
      print(f"{perc_train};{P};{R};{F1};{n_train};{n_test}")
    print()
  # exit(0)


  #
  # ablation study (dialemma -> dialemma)
  #
  for perc_train in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    P, R, F1, n_train, n_test  = eval_x_to_x("bar", "bar", tgt_dataset="dialemma", ntrees=ntrees, perc_train=perc_train, verbose=verbose)
    print(f"{perc_train};{P};{R};{F1};{n_train};{n_test}")
  # exit(0)
  
  
  #
  # dialemma -> wikidir
  # 
  src_dial = "bar"
  precisions = defaultdict(dict)
  recalls = defaultdict(dict)
  F1s = defaultdict(dict)
  for tgt_dial in tqdm.tqdm(columns):
    P, R, F1, n_train, n_test = eval_x_to_x(src_dial, tgt_dial, src_dataset="dialemma", tgt_dataset="wikidir", verbose=verbose, ntrees=ntrees, random_seed=random_seed)
    precisions["bar"][tgt_dial] = P
    recalls["bar"][tgt_dial] = R
    F1s["bar"][tgt_dial] = F1
  print("dialemma -> wikidir")
  print_results({"Precision": precisions, "Recall": recalls, "F1": F1s})
  # exit(0)


  # 
  # wikidir -> wikidir  
  #
  precisions = defaultdict(dict)
  recalls = defaultdict(dict)
  F1s = defaultdict(dict)
  combinations = [(row, col) for row in rows for col in columns]
  for src_dial, tgt_dial in tqdm.tqdm(combinations):
    P, R, F1, n_train, n_test = eval_x_to_x(src_dial, tgt_dial, src_dataset="wikidir", tgt_dataset="wikidir", verbose=verbose, ntrees=ntrees, random_seed=random_seed)
    precisions[src_dial][tgt_dial] = P
    recalls[src_dial][tgt_dial] = R
    F1s[src_dial][tgt_dial] = F1
    print(f"{src_dial},{tgt_dial}\tF1:{F1:.2f}")
  print("wikidir -> wikidir")
  print_results({"Precision": precisions, "Recall": recalls, "F1": F1s})
  # exit(0)


if __name__ == '__main__':
  main()
