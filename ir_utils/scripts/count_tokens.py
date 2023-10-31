"""

TALVEZ ADEQUAR MELHOR PENSANDO NO T5. Ver: https://github.com/huggingface/transformers/pull/24565

This script counts lines on a text file. To keep things simple, each line
will be considered a document. All other preprocessing should be done outside
of this script.
A common usage envolves removing the first column of the file (usually and id)
on queries and collections TSV files. This can be done with the following
command: `cut -f 2- queries.tsv` and passing the output of this pipe to this
script.
We also simply use tokenizer.__call__ to tokenize the documents instead of
customizing the tokenizer to the collection. This is probably not ideal for
all datasets, but is enough to get an idea of document (or query) length.
"""
import argparse
import itertools
import os

import more_itertools
import numpy as np
import pandas as pd
import tqdm
from transformers import AutoTokenizer
import ray
import glob

pd.options.display.float_format = '{:.2f}'.format

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_pattern',
                    type=str,
                    default=None,
                    help='Glob pattern to the original text files.')
parser.add_argument('--tokenizer_path',
                    type=str,
                    default=None,
                    help='Path to the tokenizer, will be loaded using '
                    'AutoTokenizer.from_pretrained')
parser.add_argument('--max_lines',
                    type=int,
                    default=None,
                    help='Max number of lines to read from the input file')
parser.add_argument('--nouse_fast_tokenizer',
                    action='store_true',
                    help='Use fast tokenizer')
parser.add_argument('--additional_tokens',
                    type=int,
                    default=0,
                    help='Number of additional tokens to add to the length '
                    'Useful for additional preprocessing or special tokens ')
parser.add_argument('--add_special_tokens',
                    action='store_true',
                    help='Add special tokens while tokenizing')
parser.add_argument('--target_length',
                    type=int,
                    default=512,
                    help='Target length of the documents.')
parser.add_argument('--output_file',
                    type=str,
                    default=None,
                    help='Path to the output file.')
parser.add_argument('--num_actors',
                    type=int,
                    default=32,
                    help='Number of actors to use to tokenize the documents.')
parser.add_argument('--batch_size',
                    type=int,
                    default=4096,
                    help='How many lines to process at once')
                    

class TokenizerActor:

    def __init__(self,
                 tokenizer_path: str,
                 use_fast: bool,
                 add_special_tokens: bool = True,
                 additional_tokens: int = 0):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       use_fast=use_fast)
        self.add_special_tokens = add_special_tokens
        self.additional_tokens = additional_tokens

    def __call__(self, batch):
        lengths = self.tokenizer(
            batch['text'].to_pylist(),
            return_length=True,
            add_special_tokens=self.add_special_tokens).length
        lengths = np.array(lengths)
        lengths += self.additional_tokens
        return {'lengths': lengths}


from ray.util.actor_pool import ActorPool
import polars as pl


def main():
    args = parser.parse_args()
    # enforce tokenizers parallelism to 1
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.verbose_progress = True
    # print(vars(ctx))
    df = pl.scan_csv(args.input_pattern,
                     has_header=False,
                     separator='\t',
                     ignore_errors=True,
                     schema={'text': pl.Utf8()})
    if args.max_lines is not None:
        df = df.head(args.max_lines)
    df = df.collect(streaming=True).rechunk()
    n_partitions = int(np.ceil(len(df) / args.batch_size))
    ds = ray.data.from_arrow(df.to_arrow())
    ds = ds.repartition(n_partitions)

    ds = ds.map_batches(TokenizerActor(
        args.tokenizer_path,
        use_fast=not args.nouse_fast_tokenizer,
        add_special_tokens=args.add_special_tokens,
        additional_tokens=args.additional_tokens),
                        compute=ray.data.ActorPoolStrategy(size=args.num_actors),
                        zero_copy_batch=True,
                        batch_format='pyarrow')
    df_token_lengths = pl.from_arrow(ray.get(ds.to_arrow_refs())).to_pandas()
    print(df_token_lengths.describe())

    describe = df_token_lengths.describe(
        percentiles=[.25, .5, .75, .9, .95, .99, .995, .999])
    print('Describe token lengths:')
    print(describe)

    # # calculate the number of tokens that will be truncated
    truncated_counts = (df_token_lengths >= args.target_length).astype(int)
    truncated_counts = truncated_counts.value_counts().to_frame('counts')
    # add percent
    truncated_counts['counts_percent'] = truncated_counts.counts.mul(100).div(
        truncated_counts.counts.sum())
    print('Truncated counts:')
    print(truncated_counts)

if __name__ == '__main__':
    main()