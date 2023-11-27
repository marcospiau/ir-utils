import json
import os
from abc import ABC
from typing import Dict, Optional, Sequence, Tuple

import patito as pt
import polars as pl
from tqdm import tqdm
from pathlib import Path
import math

from ir_utils.data_loading import load_top_k_query_doc_pairs


class BaseMappingModel(pt.Model, ABC):
    """Abstract base class for models that map a unique string to a string."""

    @classmethod
    def items_to_dataframe(cls, items: Sequence[Tuple[str,
                                                      str]]) -> pl.DataFrame:
        """Convert a sequence of (key, value) tuples to a Polars DataFrame.

        Args:
            items (Sequence[Tuple[str, str]]): A sequence of (key, value) tuples.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the items.
        """
        df = pl.DataFrame(items, schema=cls.dtypes)
        cls.validate(df)
        return df

    @classmethod
    def dict_to_dataframe(cls, items: Dict[str, str]) -> pl.DataFrame:
        """Convert a dictionary of key: value items to a Polars DataFrame.

        Args:
            items (Dict[str, str]): A dictionary of key: value items.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the items.
        """
        items = iter(items.items())
        return cls.items_to_dataframe(items)

    @classmethod
    def tsv_file_to_dataframe(cls, path: str) -> pl.DataFrame:
        """Convert a TSV file to a Polars DataFrame.

        Args:
            path (str): The path to the TSV file.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the items.
        """
        # We manually load TSV files because Polars/Pandas sometimes behave
        # inconsistently when loading TSV with newlines in the text.
        with open(path, 'r') as f:
            items = (line.strip().split('\t') for line in f)
            return cls.items_to_dataframe(items)


class QueryPTModel(BaseMappingModel):
    """Query data model."""
    query_id: str = pt.Field(unique=True)
    query: str


class DocumentPTModel(BaseMappingModel):
    """Document data model."""
    doc_id: str = pt.Field(unique=True)
    doc_text: str

    @classmethod
    def pyserini_jsonl_collection_to_dataframe(
            cls, collection_path: str) -> pl.DataFrame:
        """Load a JSONL with 'id' and 'contents' and return a Polars DataFrame.

        This is the standard data format used by Anserini/Pyserini.

        Args:
            path (str): JSONL file or a directory containing JSONL files.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the items.
        """
        collection_path = Path(collection_path)
        if collection_path.is_file():
            df = pl.scan_ndjson(collection_path)
        else:
            df = pl.concat(list(map(pl.scan_ndjson,
                                    collection_path.iterdir())))

        df = df.select('id', 'contents')
        df = df.rename({'id': 'doc_id', 'contents': 'doc_text'})
        df = df.collect()
        cls.validate(df)
        return df

    @classmethod
    def dataframe_to_pyserini_id_contents_jsonl(
            cls,
            df: pl.DataFrame,
            output_dir: str,
            n_shards: Optional[int] = None,
            max_rows_per_shard: Optional[int] = None) -> None:
        """Convert a Polars DataFrame to a JSONL with 'id' and 'contents'."""
        # only one of n_shards and max_rows_per_shard can be specified
        cls.validate(df)
        # Ensure only one of n_shards or max_rows_per_shard is specified
        if n_shards is not None and max_rows_per_shard is not None:
            raise ValueError(
                'Only one of n_shards and max_rows_per_shard can be specified')
        df = (df.rename({
            'doc_id': 'id',
            'doc_text': 'contents'
        }).select('id', 'contents'))

        # Calculate max_rows_per_shard based on n_shards, if provided
        if n_shards is not None:
            max_rows_per_shard = math.ceil(len(df) / n_shards)
        else:
            # Default to the length of the DataFrame if neither is provided
            max_rows_per_shard = max_rows_per_shard or len(df)

        # Calculate the number of output files
        n_output_files = math.ceil(len(df) / max_rows_per_shard)
        max_rows_per_shard = max_rows_per_shard or len(df)
        n_output_files = int(math.ceil(len(df) / max_rows_per_shard))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for n, chunk in tqdm(enumerate(df.iter_slices(max_rows_per_shard)),
                             total=n_output_files,
                             desc='Writing JSONL files'):
            chunk.write_ndjson(output_dir /
                               f'docs-{n:06d}-of-{n_output_files:06d}.jsonl')


class SegmentWithDocPTModel(BaseMappingModel):
    """Segment data model.

    Currently, this is only used to validate unique segment_id.
    
    """
    doc_id: str
    segment_id: str = pt.Field(unique=True)
    segment_text: str


class QueryDocumentPairPTModel(BaseMappingModel):
    """TREC run data model."""
    query_id: str
    doc_id: str

    @classmethod
    def from_trec_run(cls, path: str, top_k: int = None) -> pl.DataFrame:
        """Load a TREC run file from a specified path and return a Polars
        DataFrame.

        Args:
            path (str): The path to the TREC run file.
            top_k (int, optional): If specified, only the top k rankings will be
                included.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the run.
        """
        pairs = load_top_k_query_doc_pairs(path, top_k=top_k)
        df = pl.DataFrame(pairs, schema=cls.dtypes)
        cls.validate(df)
        return df
