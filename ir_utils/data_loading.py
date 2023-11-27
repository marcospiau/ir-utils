import glob
import json
from typing import Any, Dict, List, Optional, Set, Tuple

# from pyserini.index import IndexReader
# from pyserini.search.lucene import LuceneSearcher


def load_key_value_tsv(path: str):
    """Loads a TSV file containing key-value pairs and returns a dictionary.

    Args:
        path (str): The path to the TSV file.

    Returns:
        Dict[str, str]: A dictionary containing the key-value pairs.
    """
    data = {}
    with open(path, 'r') as f:
        for n, line in enumerate(f, 1):
            key, value = line.strip().split('\t')
            data[key] = value
    assert len(data) == n, f'Duplicate keys found in {path}'
    return data


def load_top_k_query_doc_pairs(path: str,
                               top_k: Optional[int] = None
                               ) -> List[Tuple[str, str]]:
    """Load a run file from a specified path and return a list of (qid, docid)
    tuples.

    Parameters: path (str): The path to the run file.
    top_k (int, optional): If specified, only the top k rankings will be
        included.

    Returns: List[Tuple[str, str]]: A list of (qid, docid) tuples.
    """
    run = []
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, _, _ = line.strip().split()
            rank = int(rank)
            if top_k is not None and rank > top_k:
                continue
            run.append((qid, docid))
    return run


def check_keys_match(data: Dict[str, Any], expected_keys: Set[str]) -> None:
    """Checks if the data keys match the expected set of keys.
    
    Args:
        data (Dict[str, Any]): The dictionary to check.
        expected_keys (Set[str]): The expected set of keys.
    
    Raises:
        AssertionError: If the keys in the data do not match the expected keys.
    """
    assert set(
        data.keys()) == expected_keys, 'Data keys and expected keys mismatch'


def load_filtered_jsonl_docs(ids: Set[str],
                             glob_pattern: str,
                             check_ids: bool = True) -> Dict[str, str]:
    """Loads a corpus from a set of jsonl files and returns a dictionary.

    The dictionary contains the document id as the key and the document text as
        the value.

    Args:
        ids (Set[str]): A set of document ids to be retrieved.
        glob_pattern (str): The glob pattern to match file paths.
        check_ids (bool, optional): If True, checks if the ids in the corpus
            match the provided ids. Defaults to True.

    Returns:
        Dict[str, str]: A dictionary with document ids as keys and document
            texts as values.

    Raises:
        AssertionError: If check_ids is True and the keys in the corpus do not
            match the provided ids.

    Example:
        >>> corpus = load_jsonl_corpus({'doc1', 'doc2'}, '*.jsonl')
        >>> print(corpus['doc1'])
        Document text 1
    """
    corpus = {}
    paths = glob.glob(glob_pattern)
    for path in paths:
        with open(path, 'r') as file:
            for line in file:
                doc = json.loads(line)
                if doc['id'] in ids:
                    corpus[doc['id']] = doc['contents']
    if check_ids:
        check_keys_match(corpus, ids)
    return corpus


def load_filtered_docs_from_anserini_index(
        ids: Set[str],
        index_dir: str,
        check_ids: bool = True) -> Dict[str, str]:
    """Loads a corpus from an Anserini index and returns a dictionary.

    Args:
        ids (Set[str]): A set of document ids to be retrieved.
        index_dir (str): The directory of the Anserini index.
        check_ids (bool, optional): If True, checks if the ids in the corpus
            match the provided ids. Defaults to True.

    Returns:
        Dict[str, str]: A dictionary with document ids as keys and document
            texts as values.

    Raises:
        AssertionError: If check_ids is True and the keys in the corpus do not
            match the provided ids.

    Example:
        >>> corpus = load_anserini_index_corpus({'doc1', 'doc2'},
            'index_directory')
        >>> print(corpus['doc1'])
        Document text 1
    """
    index_reader = IndexReader(index_dir)
    corpus = {}
    for doc_id in ids:
        raw = index_reader.doc(doc_id).raw()
        assert raw is not None, f'Document with id {doc_id} not found'
        loaded = json.loads(raw)
        assert loaded['id'] == doc_id, f'id mismatch for document {doc_id}'
        corpus[doc_id] = loaded['contents']
    if check_ids:
        check_keys_match(corpus, ids)
    return corpus


def load_batched_filtered_docs_from_anserini_index(
        ids: Set[str],
        index_dir: str,
        threads: int = 16,
        check_ids: bool = True) -> Dict[str, str]:
    """Loads a corpus from an Anserini index in batches and returns a
    dictionary.

    Args:
        ids (Set[str]): A set of document ids to be retrieved.
        index_dir (str): The directory of the Anserini index.
        threads (int, optional): The number of threads to use. Defaults to 16.
        check_ids (bool, optional): If True, checks if the ids in the corpus
            match the provided ids. Defaults to True.

    Returns:
        Dict[str, str]: A dictionary with document ids as keys and document
            texts as values.

    Raises:
        AssertionError: If check_ids is True and the keys in the corpus do not
            match the provided ids.

    Example:
        >>> corpus = load_anserini_index_corpus_batched({'doc1', 'doc2'},
            'index_directory')
        >>> print(corpus['doc1'])
        Document text 1
    """
    searcher = LuceneSearcher(index_dir)
    corpus = searcher.batch_doc(ids, threads=threads)
    corpus = {k: json.loads(v.raw())['contents'] for k, v in corpus.items()}
    if check_ids:
        check_keys_match(corpus, ids)
    return corpus
