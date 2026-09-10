import json
import sys
import logging
import os
import ir_measures
import random

from collections import defaultdict
from ir_measures import nDCG, R
from pyserini.search.lucene import LuceneSearcher
from pyserini.analysis import get_lucene_analyzer

from utils import PROJECT_ROOT
from utils import get_timestamp


random.seed(42)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_queries(qlang, dlang, whitelist=None):
    assert qlang == "de"
    query_ids, queries = [], []
    fName = f"{PROJECT_ROOT}/data/bm25/de.{dlang}/queries.jsonl"
    with open(fName) as f:
        for l in f:
            r = json.loads(l)
            qid = r["id"]
            if not whitelist or qid in whitelist:
                query_ids.append(qid)
                queries.append(r["contents"])
    return query_ids, queries


def load_relass(path):
    relass = {}
    with open(path, "r") as f:
        for l in f:
            r = json.loads(l)
            relass[r["src_id"]] = {did: rel for did, rel in r["tgt_results"]}
    return  relass


def save_run(run, tgt_file):
    os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
    records = []
    for qid, did2score in run.items():
        ranking = sorted([(did, score) for did, score in did2score.items()], key=lambda e: e[1], reverse=True)
        for rank, (did, score) in enumerate(ranking):
            records.append(f"{qid}\tQ0\t{did}\t{rank}\t{score}\tbm25\n")
        # records.append(json.dumps({"qid": qid, "ranking": ranking}, ensure_ascii=False) + "\n")
    with open(tgt_file, "w") as f:
        f.writelines(records)


def expand_queries(multilemma_term2variants, queries):
    expanded_queries = [
        " ".join([query] + [var for qt in query.split() for var in multilemma_term2variants.get(qt, "")])
        for query in queries
    ]
    return expanded_queries


def load_multilemma_dict(dialect):
    # multilemma
    multilemma_term2variants = {}
    fName = f"{PROJECT_ROOT}/data/multilemma/{dialect}_dictionary.jsonl"
    with open(fName) as f:
        for l in f:
            r = json.loads(l)
            multilemma_term2variants[r["term"]] = r["variants"]
    return multilemma_term2variants


def triples_to_dict(tiples):
    relass_or_run = defaultdict(dict)
    for qid, did, score_or_label in tiples:
        relass_or_run[qid][did] = score_or_label
    return relass_or_run


def search(queries, query_ids, searcher, top_k, batch_size=5):
    # Retrieve top documents from lucene index
    threads = os.cpu_count() // 2
    lucene_results = {}
    relevant_docs = []
    for i in range(0, len(queries), batch_size):
        batch_ranking = searcher.batch_search(
            queries=queries[i:i + batch_size],
            qids=[str(qid) for qid in query_ids[i:i + batch_size]],
            k=top_k,
            threads=threads
        )
        for qid, ranking in batch_ranking.items():
            lucene_results[int(qid)] = [(result.docid, result.score) for result in ranking]
            for result in ranking:
                relevant_docs.append((qid, result.docid, result.score))
    run = triples_to_dict(relevant_docs)
    return run


def search_eval_save(queries, query_ids, relass, searcher, runfile, top_k=100, ndcg_cutoff=10):
    run = search(queries, query_ids, searcher, top_k)
    result = ir_measures.calc_aggregate([nDCG @ ndcg_cutoff, R @ top_k], relass, run)
    ndcg_result = result[nDCG @ ndcg_cutoff]
    recall_at_100_result = result[R @ 100]
    print(f"nDCG@10: {round(ndcg_result, 3)}\n"
          f"Recall@100: {round(recall_at_100_result, 3)}\n")
    save_run(run, runfile)
    return ndcg_result, recall_at_100_result


def main():
    results_summary_file = f"{PROJECT_ROOT}/results/bm25_results.csv"
    if os.path.exists(results_summary_file):
        print(f"exitting, results file exists: {results_summary_file}")
        exit(0)
    
    qlang, dialects = "de", ["ksh", "nds", "als", "pfl", "bar"]
    results_runfiles_folder = f"{PROJECT_ROOT}/results/bm25_runs"
    results = []
    
    for dialect in dialects:
        multilemma_term2variants = load_multilemma_dict(dialect)

        logger.info(f"Running {qlang}->{dialect}")
        index_dir = f"{PROJECT_ROOT}/data/bm25/{qlang}.{dialect}/index/"
        # IndexReader(index_dir).stats()

        searcher = LuceneSearcher(index_dir)
        analyzer = get_lucene_analyzer(stemming=False, stopwords=False)
        searcher.set_analyzer(analyzer)

        # load relevance assessments and queries
        relass = load_relass(f"{PROJECT_ROOT}/data/bm25/de.{dialect}/analysis_variants.jsonl")
        query_ids, queries = load_queries(qlang=qlang, dlang=dialect, whitelist=set(relass.keys()))
        n_queries = len(queries)

        results_runfile = os.path.join(results_runfiles_folder, f"{qlang}-{dialect}.run")
        print(f"{qlang}.{dialect}\n")
        ndcg_result, recall_at_100_result = search_eval_save(queries, query_ids, relass, searcher, results_runfile)
        results.append((qlang, dialect, ndcg_result, recall_at_100_result, n_queries, 0, get_timestamp(), "N"))

        # expand queries with entries found in multilemma dictionary
        expanded_queries = expand_queries(multilemma_term2variants, queries)
        n_aug = sum([q == qa for q, qa in zip(queries, expanded_queries)])
        print(f"augmented: {n_aug};{len(queries)}")
        
        results_runfile = os.path.join(results_runfiles_folder, f"{qlang}-{dialect}_QE.run")
        ndcg_result, recall_at_100_result = search_eval_save(expanded_queries, query_ids, relass, searcher, results_runfile)
        results.append((qlang, dialect, ndcg_result, recall_at_100_result, n_queries, n_aug, get_timestamp(), "Y"))
    
    results = sorted(results, key=lambda elem: elem[-1])
    with open(results_summary_file, "w") as f:
        header = "qlang;dlang;nDCG@10;recall@100;n_query;n_aug;timestamp;is_aug"
        f.write(header + "\n")
        print(header)
        for qlang, dialect, ndcg_result, recall_at_100_result, n_query, n_aug, timestamp, is_aug in results:
            row = f"{qlang};{dialect};{ndcg_result:2f};{recall_at_100_result:2f};{n_query};{n_aug};{timestamp};{is_aug}"
            f.write(row + "\n")
            print(row)


if __name__ == '__main__':
    main()
