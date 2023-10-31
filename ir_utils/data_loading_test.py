import json
import tempfile
from typing import List, Optional, Tuple
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch

import patito as pt
import polars as pl
import polars.testing
import pytest
# Assuming your function is defined here
from ir_utils.data_loading import (
    check_keys_match, load_batched_filtered_docs_from_anserini_index,
    load_filtered_docs_from_anserini_index, load_filtered_jsonl_docs,
    load_top_k_query_doc_pairs, load_tsv_queries)
from ir_utils.data_models import QueryPTModel

# Update test cases to test both DataFrame and Dictionary outputs


@pytest.mark.parametrize(
    'tsv_content, expected_output_dict, expected_output_df',
    [('query1\tWhat is the capital of France?\n'
      'query2\tWhen did World War II end?', {
          'query1': 'What is the capital of France?',
          'query2': 'When did World War II end?'
      },
      pl.DataFrame({
          'qid': ['query1', 'query2'],
          'query':
          ['What is the capital of France?', 'When did World War II end?']
      })),
     ('query1\tWhat is the capital of France?', {
         'query1': 'What is the capital of France?'
     },
      pl.DataFrame({
          'qid': ['query1'],
          'query': ['What is the capital of France?']
      })), ('', {}, pl.DataFrame(schema=QueryPTModel.dtypes))])
def test_load_tsv_queries(tsv_content, expected_output_dict,
                          expected_output_df):
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmpfile:
        tmpfile.write(tsv_content)
        tmpfile.flush()

        output_dict = load_tsv_queries(tmpfile.name)
        output_dataframe = load_tsv_queries(tmpfile.name, return_df=True)
        assert output_dict == expected_output_dict
        print(output_dataframe)
        print(expected_output_df)
        pl.testing.assert_frame_equal(output_dataframe,
                                      expected_output_df,
                                      check_row_order=False)


def test_load_tsv_queries_duplicate_keys():
    tsv_content = 'query1\tWhat is the capital of France?\n' \
                  'query1\tWhen did World War II end?'

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmpfile:
        tmpfile.write(tsv_content)
        tmpfile.flush()

        with pytest.raises(pt.ValidationError):
            load_tsv_queries(tmpfile.name)


@pytest.mark.parametrize(
    'run_content, top_k, expected_output',
    [
        # Test without top_k
        ('query1 Q0 doc1 1 100 run1\n'
         'query1 Q0 doc2 2 90 run1\n'
         'query1 Q0 doc3 3 80 run1\n'
         'query2 Q0 doc4 1 95 run1\n'
         'query2 Q0 doc5 2 85 run1', None, [('query1', 'doc1'),
                                            ('query1', 'doc2'),
                                            ('query1', 'doc3'),
                                            ('query2', 'doc4'),
                                            ('query2', 'doc5')]),

        # Test with top_k=2
        ('query1 Q0 doc1 1 100 run1\n'
         'query1 Q0 doc2 2 90 run1\n'
         'query1 Q0 doc3 3 80 run1\n'
         'query2 Q0 doc4 1 95 run1\n'
         'query2 Q0 doc5 2 85 run1', 2, [('query1', 'doc1'),
                                         ('query1', 'doc2'),
                                         ('query2', 'doc4'),
                                         ('query2', 'doc5')]),

        # Test with an empty file
        ('', None, [])
    ])
def test_load_top_k_query_doc_pairs(run_content, top_k, expected_output):
    m = mock_open(read_data=run_content)
    with patch('builtins.open', m):
        assert load_top_k_query_doc_pairs('fake_path',
                                          top_k) == expected_output


def test_invalid_file_load_top_k_query_doc_pairs():
    # Mocking an invalid run file content
    invalid_content = 'invalid_content'
    m = mock_open(read_data=invalid_content)
    with patch('builtins.open', m):
        with pytest.raises(ValueError):
            load_top_k_query_doc_pairs('fake_path')


@pytest.mark.parametrize(
    'data, expected_keys, error_message',
    [({
        'key1': 'value1',
        'key2': 'value2'
    }, {'key1', 'key2'}, None),
     ({
         'key1': 'value1',
         'key2': 'value2'
     }, {'key1', 'key3'}, 'Data keys and expected keys mismatch')])
def test_check_keys_match(data, expected_keys, error_message):
    if error_message:
        with pytest.raises(AssertionError, match=error_message):
            check_keys_match(data, expected_keys)
    else:
        check_keys_match(data, expected_keys)


def test_load_filtered_jsonl_docs_no_check():
    ids = {'doc1', 'doc2'}
    mock_data = ('{"id": "doc1", "contents": "Document text 1"}\n'
                 '{"id": "doc2", "contents": "Document text 2"}')
    m = mock_open(read_data=mock_data)

    with patch('builtins.open', m), patch('glob.glob',
                                          return_value=['mock_file.jsonl']):
        corpus = load_filtered_jsonl_docs(ids, '*.jsonl', check_ids=False)
        assert corpus == {'doc1': 'Document text 1', 'doc2': 'Document text 2'}


def create_mock_doc(doc_id, text):
    """Helper function to create a mock document."""
    mock_doc = mock.MagicMock()
    mock_doc.raw.return_value = json.dumps({'id': doc_id, 'contents': text})
    return mock_doc


def test_load_filtered_docs_from_anserini_index_with_valid_data():
    mock_index = mock.MagicMock()

    # Create mock documents for 10 docs
    num_docs = 10
    doc_ids = [f'doc{i}' for i in range(1, num_docs + 1)]
    mock_docs = {
        doc_id: create_mock_doc(doc_id, f'Document text {i}')
        for i, doc_id in enumerate(doc_ids, 1)
    }

    def side_effect(doc_id):
        return mock_docs[doc_id]

    mock_index.doc.side_effect = side_effect

    with mock.patch('ir_utils.data_loading.IndexReader', return_value=mock_index), \
         mock.patch('ir_utils.data_loading.check_keys_match'):

        result = load_filtered_docs_from_anserini_index(
            set(doc_ids), 'dummy_index_dir')

        for i, doc_id in enumerate(doc_ids, 1):
            assert result[doc_id] == f'Document text {i}'


def test_load_batched_filtered_docs_from_anserini_index_with_valid_data():
    mock_searcher = mock.MagicMock()

    # Create mock documents for 10 docs
    num_docs = 10
    doc_ids = [f'doc{i}' for i in range(1, num_docs + 1)]
    mock_docs = {
        doc_id: create_mock_doc(doc_id, f'Document text {i}')
        for i, doc_id in enumerate(doc_ids, 1)
    }

    mock_searcher.batch_doc.return_value = mock_docs

    with mock.patch('ir_utils.data_loading.LuceneSearcher', return_value=mock_searcher), \
         mock.patch('ir_utils.data_loading.check_keys_match'):

        result = load_batched_filtered_docs_from_anserini_index(
            set(doc_ids), 'dummy_index_dir')

        for i, doc_id in enumerate(doc_ids, 1):
            assert result[doc_id] == f'Document text {i}'
