import sys
import os
import shutil

from functools import partial
from huggingface_hub import hf_hub_download 


local_dir = sys.argv[1]

repo_id = "rlitschk/wikidir"
os.makedirs(local_dir, exist_ok=True)
download_fn = partial(hf_hub_download,
  repo_id=repo_id, 
  repo_type="dataset", 
  local_dir=local_dir,
  local_dir_use_symlinks=False
)

for dial in ["bar", "als", "ksh", "pfl", "nds"]:
  print(f"downloading from de.{dial}/")
  download_fn(filename=f"de.{dial}/queries.jsonl")
  download_fn(filename=f"de.{dial}/docs.jsonl")
  download_fn(filename=f"de.{dial}/analysis_variants.jsonl")
  index_dir = os.path.join(local_dir, f"de.{dial}", "index")
  os.makedirs(index_dir, exist_ok=True)
  shutil.move(
    os.path.join(local_dir, f"de.{dial}", "docs.jsonl"), 
    os.path.join(index_dir, "docs.jsonl")
  )
