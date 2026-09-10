import os
import csv
import random
import pandas as pd

# random.seed(0)

from utils import PROJECT_ROOT
from utils import parse_args
from utils import postprocess_recognition
from eval import evaluate
from eval import eval_x_to_x
from utils import get_timestamp


args = parse_args()
random.seed(args.random_seed)
SEP=";"
percentage_cutoffs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 0.0001, 0.001,


def maybe_exit(results_file):
  if os.path.exists(results_file):
    with open(results_file) as f:
      next(f)
      for r in f:
        value = r.split(";")[-2]
        if value == "AVG":
          exit(0)
        processed_rand_seed = int(value)
        if processed_rand_seed == args.random_seed:
          print(f"exitting, random_seed={args.random_seed} already processed.")
          exit(0)


def run_baselines():
  with open(f"{PROJECT_ROOT}/data/dialemma/mistral-large.csv") as file:
    reader = csv.DictReader(
      file,
      fieldnames= [
        "id", "lemma_id", "de_term", "de_freq", "pos", "pos_perc", "bar", "bar_freq","ld", "label", "contexts", "prediction"
      ]
    )
    next(reader)
    y_hat_mistral = []
    y_hat_random = []
    y = []
    for r in reader:
      # 80K corresponds to 80% of all records / lines in `dialemma_bar.jsonl` (=`recognition.csv` in dialemma) 
      # see see scripts/download_dialemma_files.sh
      if int(r["id"]) >= 80_000:
        y_hat_mistral.append(postprocess_recognition(r["prediction"]))
        y_hat_random.append(random.choice(["yes", "inflected", "no"]))
        y.append(r["label"])

  results = []
  P, R, F1 = evaluate(y_hat=y_hat_random, y_test=y, label="yes")
  results.append(SEP.join(["random", str(P), str(R), str(F1), str(args.random_seed), get_timestamp()]))

  P, R, F1 = evaluate(y_hat=y_hat_mistral, y_test=y, label="yes")
  results.append(SEP.join(["mistral-large", str(P), str(R), str(F1), str(args.random_seed), get_timestamp()]))

  return results


def run_main():
  results_folder = "results"
  results_file = os.path.join(results_folder, "dialemma_main.csv")
  os.makedirs(results_folder, exist_ok=True)
  results_file_exists = os.path.exists(results_file)
  # maybe_exit(results_file)

  # mistral-large baseline on lrec test set (80% of all data)
  results = run_baselines()

  # Random forest results
  P, R, F1, n_train, n_test  = eval_x_to_x(
    src_dialect="bar",
    tgt_dialect="bar",
    src_dataset="dialemma",
    tgt_dataset="dialemma",
    ntrees=args.ntrees,
    verbose=args.verbose,
    random_seed=args.random_seed
  )
  results.append(SEP.join(["random_forest", str(P), str(R), str(F1), str(args.random_seed), get_timestamp()]))

  with open(results_file, "a" if results_file_exists else "w") as f:
    if not results_file_exists:
      header = "model;precision;recall;f1;random_seed;timestamp"
      f.write(header + "\n")
      print(header)
    for row in results:
      f.write(row + "\n")
      print(row)


def run_ablation():
  results_file = f"{PROJECT_ROOT}/results/dialemma_ablation.csv"
  results_file_exists = os.path.exists(results_file)
  maybe_exit(results_file)

  results = []
  for perc_train in percentage_cutoffs:
    P, R, F1, n_train, n_test  = eval_x_to_x(
      src_dialect="bar",
      tgt_dialect="bar",
      tgt_dataset="dialemma",
      ntrees=args.ntrees,
      perc_train=perc_train,
      verbose=args.verbose,
      random_seed=args.random_seed
    )
    row = SEP.join([
      "random_forest", str(perc_train), str(P), str(R), str(F1), str(n_train), str(n_test),
      str(args.random_seed), get_timestamp()
    ])
    results.append(row)
    print(f"{perc_train};{P};{R};{F1};{n_train};{n_test}")

  with open(results_file, "a" if results_file_exists else "w") as f:
    if not results_file_exists:
      header = SEP.join(["model","perc_train","precision","recall","f1","n_train","n_test","random_seed","timestamp"])
      f.write(header + "\n")
      print(header)
    for row in results:
      f.write(row + "\n")
      print(row)


def aggregate_seeds_ablation(filename):
  if os.path.exists(filename):
    df = pd.read_csv(filename, sep=SEP)
    if "AVG" not in set(df["random_seed"].tolist()):
      for cutoff in percentage_cutoffs[::-1]:
        cutoff_filter = df["perc_train"] == cutoff
        grouped_df = df[cutoff_filter].groupby('model').agg(
          {
            'perc_train': 'first',
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean',
            'n_train': 'first',
            'n_test': 'first',
            'random_seed': lambda x: 'AVG',
            'timestamp':  lambda x: get_timestamp()
          }
        )
        grouped_df = grouped_df.reset_index()
        df = pd.concat([grouped_df, df], ignore_index=True)
      df.to_csv(filename, sep=SEP, index=False)


def aggregate_seeds_main(filename):
  if os.path.exists(filename):
    df = pd.read_csv(filename, sep=SEP)
    if "AVG" not in set(df["random_seed"].tolist()):
      for model in ["random_forest", "mistral-large", "random"]:
        model_filter = df["model"] == model
        grouped_df = df[model_filter].groupby('model').agg(
          {
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean',
            'random_seed': lambda x: 'AVG',
            'timestamp': lambda x: get_timestamp()
          }
        )
        grouped_df = grouped_df.reset_index()
        df = pd.concat([grouped_df, df], ignore_index=True)
      df.to_csv(filename, sep=SEP, index=False)


if __name__ == '__main__':
  if args.experiment == "main":
    run_main()

  elif args.experiment == "main_summarize":
    aggregate_seeds_main(filename=f"{PROJECT_ROOT}/results/dialemma_main.csv")

  elif args.experiment == "ablation":
    run_ablation()

  elif args.experiment == "ablation_summarize":
    aggregate_seeds_ablation(filename=f"{PROJECT_ROOT}/results/dialemma_ablation.csv")
