import os
import time
import datetime 
from argparse import ArgumentParser


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # get parent directry

out_fieldnames = ["id", "lemma_id", "de_term", "de_freq", "pos", "pos_perc", "dial", "dial_freq", "ld", "label", "contexts"]


def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')


def postprocess_recognition(prediction):
  prediction = prediction.replace("```", "").replace("plaintext", "").strip()
  prediction = prediction.lower()
  prediction = "".join([c if c.isalpha() else " " for c in prediction]).strip()
  prediction_toks = prediction.split()
  if prediction_toks:
    prediction = prediction_toks[0]
  return prediction


def rm_begriffserklaerung(term):
  if "(" in term:
    rm_substring = term[term.index("("):term.index(")")+1]
    return term.replace(rm_substring, "").strip()
  else:
    return term
  
def parse_args():
  parser = ArgumentParser()
  parser.add_argument("--random_seed", type=int, help="Random seed", default=1, required=False)
  parser.add_argument("--ntrees", type=int, help="Number of trees", default=100, required=False)
  parser.add_argument("--experiment", type=str, help="main experiment or ablation", required=False, choices=["main", "main_summarize", "ablation", "ablation_summarize"], default="main")
  parser.add_argument("--verbose", type=bool, default=False, help="Verbose output", required=False)
  return parser.parse_args()
