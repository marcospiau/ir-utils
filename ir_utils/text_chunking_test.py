import sys
from typing import List
from unittest.mock import patch

import pytest
import spacy
from ir_utils.text_chunking import (
    add_segment_to_doc_id, chunk_corpus, chunk_document_into_sentences,
    chunk_sentences, deserialize_spacy_pipeline, get_sentences_from_doc,
    initialize_spacy_pipeline, serialize_spacy_pipeline)


def test_add_segment_to_doc_id():
    assert add_segment_to_doc_id('doc1', 1) == 'doc1#1'
    assert add_segment_to_doc_id('doc1', 2, '-') == 'doc1-2'
    with pytest.raises(AssertionError):
        add_segment_to_doc_id('doc1#already', 1)


def test_initialize_spacy_pipeline():
    nlp = initialize_spacy_pipeline('en')
    assert isinstance(nlp, spacy.language.Language)
    assert nlp.max_length == sys.maxsize


def test_serialization_and_deserialization():
    nlp = initialize_spacy_pipeline('en')
    config, bytes_data = serialize_spacy_pipeline(nlp)
    nlp2 = deserialize_spacy_pipeline(config, bytes_data)
    assert nlp.meta == nlp2.meta


def test_get_sentences_from_doc():
    nlp = initialize_spacy_pipeline('en')
    sentences = get_sentences_from_doc('This is a test. Another test.', nlp)
    assert len(sentences) == 2
    assert sentences == ['This is a test.', 'Another test.']


@pytest.mark.parametrize(
    'sentences, stride, max_length, expected_chunks',
    [
        # Test where the entire list is one chunk
        ([
            'This is a test.', 'Another test.', 'Yet another test.',
            'Still going.', 'And another one.', 'Almost done.', 'Last one.'
        ], 5, 10, [
            ' '.join([
                'This is a test.', 'Another test.', 'Yet another test.',
                'Still going.', 'And another one.', 'Almost done.', 'Last one.'
            ])
        ]),

        # Test where sentences are split into multiple chunks based on stride
        #   and max_length
        ([
            'This is a test.', 'Another test.', 'Yet another test.',
            'Still going.', 'And another one.', 'Almost done.', 'Last one.'
        ], 3, 4, [
            ' '.join([
                'This is a test.', 'Another test.', 'Yet another test.',
                'Still going.'
            ]), ' '.join([
                'Still going.', 'And another one.', 'Almost done.', 'Last one.'
            ])
        ]),

        # Add more test cases as needed...
    ])
def test_chunk_sentences(sentences: List[str], stride: int, max_length: int,
                         expected_chunks: List[str]):
    chunks = list(chunk_sentences(sentences, stride, max_length))
    assert len(chunks) == len(expected_chunks)
    for c, e in zip(chunks, expected_chunks):
        assert c == e


def test_chunk_corpus():
    corpus = [('doc1', 'This is a test. Another test.'),
              ('doc2', 'Yet another test. Still going. And another one.')]
    chunks = chunk_corpus(corpus, 'en', show_progress=False)
    assert len(chunks) == 2
    assert len(chunks['doc1']) == 1
    assert len(chunks['doc2']) == 1
    assert chunks['doc1'][0][1] == 'This is a test. Another test.'
    assert chunks['doc2'][0][
        1] == 'Yet another test. Still going. And another one.'