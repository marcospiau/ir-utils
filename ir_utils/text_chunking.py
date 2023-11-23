import functools
import sys
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import polars as pl
import spacy
from ir_utils.data_models import DocumentPTModel, SegmentWithDocPTModel
from spacy.language import Language
from tqdm import tqdm
import pyarrow as pa

DOCUMENT_SEGMENT_DELIMITER = '#'


def add_segment_to_doc_id(
        doc_id: str,
        segment_id: int,
        segment_delimiter: str = DOCUMENT_SEGMENT_DELIMITER) -> str:
    """Adds a delimiter to a document id.

    Args:
        doc_id (str): The document id.
        segment_id (int): The segment id.

    Returns:
        str: The document id with the delimiter.
    """
    assert segment_delimiter not in doc_id
    return f'{doc_id}{segment_delimiter}{segment_id}'


def initialize_spacy_pipeline(language: str, max_length=None) -> Language:
    """ Initialize a spaCy pipeline.

    Args:
        language (str): The language of the pipeline.
        max_length (int, optional): The maximum length of the pipeline.
            Defaults to None, which will use sys.maxsize.

    Returns:
        Language: The initialized spaCy pipeline.
    """

    nlp = spacy.blank(language)
    nlp.add_pipe('sentencizer')
    nlp.max_length = max_length or sys.maxsize
    return nlp


def serialize_spacy_pipeline(nlp: Language) -> Tuple[dict, bytes]:
    """Serialize a spaCy pipeline.

    Args:
        nlp (Language): The spaCy pipeline to be serialized.

    Returns:
        Tuple[dict, bytes]: A tuple containing the config dict and serialized
            bytes data.

    Example:
        >>> config, bytes_data = serialize_spacy_pipeline(nlp)
    """
    config = nlp.config
    bytes_data = nlp.to_bytes()
    return config, bytes_data


def deserialize_spacy_pipeline(config: dict, bytes_data: bytes) -> Language:
    """Deserialize a spaCy pipeline.

    Args:
        config (dict): The configuration dictionary.
        bytes_data (bytes): The serialized bytes data.

    Returns:
        Language: The deserialized spaCy pipeline.

    Example:
        >>> nlp = deserialize_spacy_pipeline(config, bytes_data)
    """
    lang_cls = spacy.util.get_lang_class(config['nlp']['lang'])
    nlp = lang_cls.from_config(config)
    nlp.from_bytes(bytes_data)
    return nlp


def get_sentences_from_doc(doc_text: str, nlp: Language) -> List[str]:
    """Splits a document into sentences.

    Args:
        doc_text (str): text of the document
        nlp (Language): spacy pipeline

    Returns:
        List[str]: list of sentences
    """
    doc = nlp(doc_text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def chunk_sentences(sentences: List[str],
                    stride=5,
                    max_length=10) -> Iterator[str]:
    """Chunk sentences in windows.

    Args:
        sentences (List[str]): list of sentences
        stride (int, optional): window stride. Defaults to 5.
        max_length (int,optional): max window length. Defaults to 10.

    Yields:
        Iterator[str]: iterator over the chunks
    """
    for i in range(0, len(sentences), stride):
        segment = ' '.join(sentences[i:i + max_length])
        yield segment
        if i + max_length >= len(sentences):
            break


def chunk_document_into_sentences(
    doc_id: str,
    doc_text: str,
    nlp: Language,
    max_doc_char_length: Optional[int] = None,
    window_stride: int = 5,
    window_max_length: int = 10,
    docid_segment_delimiter: str = DOCUMENT_SEGMENT_DELIMITER
) -> List[Tuple[str, str]]:
    """
    Breaks down a document into smaller segments.

    Args:
        doc_id (str): The ID of the document.
        doc_text (str): The text of the document.
        nlp (Language): The spaCy pipeline.
        max_doc_char_length (Optional[int]): Max characters allowed in a document.
        window_stride (int): Number of sentences to advance when creating chunks.
        window_max_length (int): Max sentences in each chunk.
        docid_segment_delimiter (str): Delimiter to separate document ID from segment ID.

    Returns:
        List[Tuple[str, str]]: List of segments. Each segment is represented as a tuple
        containing a modified document ID and segment content.
    """
    if max_doc_char_length is not None:
        doc_text = doc_text[:max_doc_char_length]
    sentences = get_sentences_from_doc(doc_text, nlp)
    sentence_chunks = chunk_sentences(sentences,
                                      stride=window_stride,
                                      max_length=window_max_length)
    return [(add_segment_to_doc_id(doc_id=doc_id,
                                   segment_id=chunk_id,
                                   segment_delimiter=docid_segment_delimiter),
             chunk) for chunk_id, chunk in enumerate(sentence_chunks)]


def chunk_corpus_with_ray(
        corpus_items_dataframe: pl.DataFrame, parallelism: int,
        **chunk_corpus_kwargs) -> Dict[str, List[Tuple[str, str]]]:
    """Breaks down a corpus into smaller document segments using Ray."""
    import ray
    ds = ray.data.from_arrow(corpus_items_dataframe.to_arrow())
    ds = ds.repartition(parallelism)
    ds = ds.map_batches(fn=lambda df: pd.DataFrame(
        chunk_corpus(corpus_items=df.itertuples(index=False),
                     **chunk_corpus_kwargs).items(),
        columns=['docid', 'chunks']),
                        zero_copy_batch=True,
                        batch_format='pandas')
    ds = pl.from_arrow(ray.get(ds.to_arrow_refs()))
    ray.shutdown()
    return ds


class CorpusChunker:
    """Breaks down a corpus into smaller document segments.

        language (str): The language of the documents in the corpus.
        max_doc_char_length (Optional[int]): Max characters allowed in a
            document. Defaults to sys.maxsize.
        window_stride (int, default=5): Number of sentences to advance when
            creating chunks.
        window_max_length (int, default=10): Max sentences in each chunk.
        show_progress (bool, default=False): Display a progress bar.
        docid_segment_delimiter (str, default=DOCUMENT_SEGMENT_DELIMITER):
            Delimiter to separate document ID from segment ID.
    """

    def __init__(self,
                 language: str,
                 max_doc_char_length: Optional[int] = None,
                 window_stride: int = 5,
                 window_max_length: int = 10,
                 docid_segment_delimiter: str = DOCUMENT_SEGMENT_DELIMITER):
        self.language = language
        self.max_doc_char_length = max_doc_char_length or sys.maxsize
        self.window_stride = window_stride
        self.window_max_length = window_max_length
        self.docid_segment_delimiter = docid_segment_delimiter
        self.chunk_document_into_sentences_partial = functools.partial(
            chunk_document_into_sentences,
            max_doc_char_length=self.max_doc_char_length,
            window_stride=self.window_stride,
            window_max_length=self.window_max_length,
            docid_segment_delimiter=self.docid_segment_delimiter)

    @property
    def nlp(self) -> Language:
        # this is not an attribute to avoid serialization
        return initialize_spacy_pipeline(self.language,
                                         self.max_doc_char_length)

    def chunk_corpus_items(
            self,
            corpus_items,
            show_progress: bool = True) -> Dict[str, List[Tuple[str, str]]]:
        return {
            doc_id:
            self.chunk_document_into_sentences_partial(doc_id=doc_id,
                                                       doc_text=doc_text,
                                                       nlp=self.nlp)
            for doc_id, doc_text in tqdm(corpus_items,
                                         disable=not show_progress)
        }

    def chunk_corpus_polars(self, df_corpus: pl.DataFrame) -> pl.DataFrame:
        nlp = self.nlp
        chunk_expr = pl.struct('doc_id', 'doc_text').map_elements(
            lambda x: self.chunk_document_into_sentences_partial(
                **x, nlp=nlp)).alias('segments')
        df_chunks = df_corpus.select('doc_id', chunk_expr).explode('segments')
        df_chunks = df_chunks.select(
            'doc_id',
            pl.col('segments').list.to_struct(
                fields=['segment_id', 'segment_text'])).unnest('segments')
        SegmentWithDocPTModel.validate(df_chunks)
        return df_chunks

    def chunk_corpus_arrow(self, pa_table: pa.Table) -> pa.Table:
        df = pl.from_arrow(pa_table)
        return self.chunk_corpus_polars(df).to_arrow()

    def chunk_corpus_ray(self,
                         df_corpus: pl.DataFrame,
                         ray_parallelism: int = 1,
                         ray_shutdown: bool = True) -> pl.DataFrame:
        import ray
        ray.data.DataContext.get_current(
        ).execution_options.verbose_progress = True
        ds = ray.data.from_arrow(df_corpus.to_arrow())
        ds = ds.repartition(ray_parallelism)
        # triggering the reparition computation
        _ = ds.count()
        ds = ds.map_batches(self.chunk_corpus_arrow,
                            zero_copy_batch=True,
                            batch_format='pyarrow')
        df_chunks = pl.from_arrow(ray.get(ds.to_arrow_refs()))
        SegmentWithDocPTModel.validate(df_chunks)
        if ray_shutdown:
            ray.shutdown()
        return df_chunks
