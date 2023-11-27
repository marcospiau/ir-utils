"""This chunks a collection into smaller documents using spaCy sentence
tokenizer.
"""
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import polars as pl

from ir_utils.data_models import DocumentPTModel, SegmentWithDocPTModel
from ir_utils.text_chunking import CorpusChunker

# Set up basic configuration for the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(pathname)s] %(message)s')

# Create a logger instance with a specific name
logger = logging.getLogger(__name__)

parser = ArgumentParser(description='Chunk jsonl collection',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Corpus Loading Arguments
input_data_group = parser.add_argument_group('Input Data')
input_data_group.add_argument(
    '--input_collection',
    help='Path to a jsonl files or a directory containing jsonl files')
# Output Arguments
out_group = parser.add_argument_group('Output')
out_group.add_argument('--output_dir',
                       type=str,
                       required=True,
                       help='Output directory for monoT5 input files')
sharding_mutually_exclusive_group = out_group.add_mutually_exclusive_group(
    required=False)
sharding_mutually_exclusive_group.add_argument(
    '--n_shards',
    type=int,
    required=False,
    default=None,
    help='The number of output shards to write. '
    'Defaults to None (single shard).')
sharding_mutually_exclusive_group.add_argument(
    '--max_rows_per_shard',
    type=int,
    required=False,
    default=None,
    help='The maximum number of rows per shard. '
    'Defaults to None (single shard).')

# Document Splitting Arguments
doc_split_group = parser.add_argument_group('Document Splitting')
doc_split_group.add_argument('--language',
                             type=str,
                             required=True,
                             help='Language for the Spacy sentence tokenizer')
doc_split_group.add_argument(
    '--max_length',
    type=int,
    default=10,
    help='Maximum number of sentences in each segment (default: 10)')
doc_split_group.add_argument(
    '--stride',
    type=int,
    default=5,
    help='Stride (step) in sentences between each segment (default: 5)')
doc_split_group.add_argument(
    '--max_doc_char_length',
    type=int,
    default=None,
    help='Maximum number of characters allowed in a document. '
    'If None (default), will use sys.maxsize')

# Parallel Processing Arguments
parallel_processing_group = parser.add_argument_group('Parallel Processing')
parallel_processing_group.add_argument(
    '--use_ray',
    action='store_true',
    help='Use Ray for parallel processing (default: False)')
parallel_processing_group.add_argument(
    '--ray_parallelism',
    type=int,
    default=1,
    help='Number of parallel processes (default: 1)')


def main():
    args = parser.parse_args()
    logging.info('args: %s', args)

    corpus_chunker = CorpusChunker(
        language=args.language,
        max_doc_char_length=args.max_doc_char_length,
        window_stride=args.stride,
        window_max_length=args.max_length)
    df_chunks = DocumentPTModel.pyserini_jsonl_collection_to_dataframe(
        args.input_collection)
    logging.info('Loaded %d documents from %s', len(df_chunks),
                 args.input_collection)
    if args.use_ray is True:
        logger.info('Chunking corpus using Ray')
        df_chunks = corpus_chunker.chunk_corpus_ray(
            df_chunks, ray_parallelism=args.ray_parallelism)
    else:
        logger.info('Chunking corpus using Polars (single process)')
        df_chunks = corpus_chunker.chunk_corpus_polars(df_chunks)

    SegmentWithDocPTModel.validate(df_chunks)
    # convert to DocumentPTModel
    df_chunks = df_chunks.select(
        pl.col('segment_id').alias('doc_id'),
        pl.col('segment_text').alias('doc_text'))
    logging.info('Writing to %s', args.output_dir)
    DocumentPTModel.dataframe_to_pyserini_id_contents_jsonl(
        df=df_chunks,
        output_dir=args.output_dir,
        n_shards=args.n_shards,
        max_rows_per_shard=args.max_rows_per_shard)
    logging.info('Done')


if __name__ == '__main__':
    main()
