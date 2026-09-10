import random
import multiprocessing

from collections import Counter
from functools import partial
from featurize import featurize
from dataloader import get_raw_dataset
from sklearn.ensemble import RandomForestClassifier


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
        dataset_generator = get_raw_dataset(dataset, dialect, split)
        for xi, yi in p.imap(fn, dataset_generator):
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


def eval_x_to_x(src_dialect, tgt_dialect, src_dataset="dialemma", tgt_dataset="dialemma",
                verbose=False, ntrees=100, perc_train=1.0, perc_test=1.0, random_seed=1):
  """
  This method loads the data, computes features, trains a random forest, generates predictions and evaluates result. 
  
  :param src_dialect: source dialect
  :param tgt_dialect: target dialect
  :param src_dataset: source dataset
  :param tgt_dataset: target dataset
  :param verbose: print intermediate outputs
  :param ntrees: number of trees in random forest
  :param perc_train: Percentage of training data used for training
  :param perc_test: Percentage of test data used for testing
  :param random_seed: random seed
  :return: 
  """
  if tgt_dialect != "bar" and tgt_dataset == "dialemma":
    if verbose: print("dialemma only possible for bar")
    return -1, -1, -1

  if verbose: print("featurizing train")
  
  # load training data
  if src_dataset == "dialemma": assert src_dialect == "bar"
  
  x_train, y_train = get_featurized_dataset(dataset=src_dataset, dialect=src_dialect, split="train")
  indices = list(range(len(x_train)))

  # shuffle training set
  random.seed(random_seed)
  random.shuffle(indices)
  x_train = [x_train[i] for i in indices]
  y_train = [y_train[i] for i in indices]

  # take subset according to the specified percentage of the training data
  n_train = max(1, int(perc_train * len(x_train)))
  x_train, y_train = x_train[:n_train], y_train[:n_train]
  
  x_test, y_test = get_featurized_dataset(dataset=tgt_dataset, dialect=tgt_dialect, split="test")
  n_test = max(1, int(perc_test * len(x_test)))
  x_test, y_test = x_test[:n_test], y_test[:n_test]

  # train model
  clf = RandomForestClassifier(
    n_estimators=ntrees, 
    n_jobs=10, 
    random_state=random_seed, 
    verbose=verbose,
  ).fit(x_train, y_train)

  P, R, F1_yes, freqs = predict_and_evaluate(model=clf, x_test=x_test, y_test=y_test, verbose=verbose)
  return P, R, F1_yes, n_train, n_test
