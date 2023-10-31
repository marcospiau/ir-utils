import patito as pt
from typing import Dict, List, Sequence, Tuple
import polars as pl
from abc import ABC
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


class SegmentWithDocPTModel(BaseMappingModel):
    """Segment data model."""
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
