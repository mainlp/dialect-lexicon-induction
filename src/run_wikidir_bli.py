import os
import tqdm
import pandas as pd
from collections import defaultdict
from eval import eval_x_to_x
from utils import PROJECT_ROOT
from utils import parse_args


args = parse_args()

# source dialects
rows = ["ksh", "nds", "als", "pfl", "bar", "ALL"]
# target dialects
columns = ["ksh", "nds", "als", "pfl", "bar"]

SEP = ";"


def update_results(all_results, measures):
  for m in measures:
    results = all_results[m]
    print(m)
    results_file = f"{PROJECT_ROOT}/results/wikidir_main_{m}.csv"
    file_exists = os.path.exists(results_file)
    with open(results_file, "a" if file_exists else "w") as f:
      header = SEP.join([f"src\\tgt"] + columns + ["random_seed"])
      print(header)
      if not file_exists:
        f.write(header + "\n")
      for donor in results.keys():
        csv_row = SEP.join(
          [donor] + [str(results[donor][recepient]) for recepient in columns] + [str(args.random_seed)])
        print(csv_row)
        f.write(csv_row + "\n")


def run_ablation():
  for dial in rows:
    print(dial)
    print("P;R;F1;n_train;n_test")
    for perc_train in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
      P, R, F1, n_train, n_test  = eval_x_to_x(
        src_dialect="ALL", 
        tgt_dialect=dial, 
        src_dataset="wikidir", 
        tgt_dataset="wikidir", 
        ntrees=args.ntrees, 
        random_seed=args.random_seed, 
        perc_train=perc_train, 
        verbose=args.verbose
      )
      print(f"{perc_train};{P};{R};{F1};{n_train};{n_test}")
    print()


def run_main():
  measures = ["precision", "f1", "recall"]
  precisions = defaultdict(dict)
  recalls = defaultdict(dict)
  F1s = defaultdict(dict)
  combinations = [(row, col) for row in rows for col in columns]
  for src_dial, tgt_dial in tqdm.tqdm(combinations):
    P, R, F1, n_train, n_test = eval_x_to_x(
      src_dial, 
      tgt_dial, 
      src_dataset="wikidir", 
      tgt_dataset="wikidir", 
      verbose=args.verbose, 
      ntrees=args.ntrees, 
      random_seed=args.random_seed
    )
    precisions[src_dial][tgt_dial] = P
    recalls[src_dial][tgt_dial] = R
    F1s[src_dial][tgt_dial] = F1
    print(f"{src_dial},{tgt_dial}\tF1:{F1:.2f}")
  print("wikidir -> wikidir")
  all_results = {"precision": precisions, "recall": recalls, "f1": F1s}
  update_results(all_results, measures)


def aggregate_seeds_main(filename):
  if os.path.exists(filename):
    df = pd.read_csv(filename, sep=SEP)
    if "AVG" not in set(df["random_seed"].tolist()):
      for donor in ["ksh", "nds", "als", "pfl", "bar", "ALL"][::-1]:
        donor_filter = df["src\\tgt"] == donor
        grouped_df = df[donor_filter].groupby("src\\tgt").agg(
          {
            'ksh': 'mean',
            'nds': 'mean',
            'als': 'mean',
            'pfl': 'mean',
            'bar': 'mean',
            'random_seed': lambda _: 'AVG'
          }
        )
        grouped_df = grouped_df.reset_index()
        df = pd.concat([grouped_df, df], ignore_index=True)
      df.to_csv(filename, sep=SEP, index=False)


if __name__ == '__main__':
  if args.experiment == "main":
    run_main()
  
  elif args.experiment == "main_summarize":
    for m in ["precision", "recall", "f1"]:
      aggregate_seeds_main(filename=f"results/wikidir_main_{m}.csv")
  
  elif args.experiment == "ablation":
    run_ablation()

  elif args.experiment == "ablation_summarize":
    raise NotImplementedError
