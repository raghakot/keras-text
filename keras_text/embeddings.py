from __future__ import absolute_import

import os
import logging
import numpy as np
from keras.utils.data_utils import get_file


logger = logging.getLogger(__name__)
_EMBEDDINGS_CACHE = dict()

# Add more types here as needed.
_EMBEDDING_TYPES = {
    'glove.42B.300d': {
        'file': 'glove.42B.300d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
    },

    'glove.6B.50d': {
        'file': 'glove.6B.50d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.6B.zip'
    },

    'glove.6B.100d': {
        'file': 'glove.6B.100d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.6B.zip'
    },

    'glove.6B.200d': {
        'file': 'glove.6B.200d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.6B.zip'
    },

    'glove.6B.300d': {
        'file': 'glove.6B.300d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.6B.zip'
    },

    'glove.840B.300d': {
        'file': 'glove.840B.300d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    }
}


def _build_embeddings_index(embeddings_path):
    logger.info('Building embeddings index...')
    index = {}
    with open(embeddings_path, 'rb') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            index[word] = vector
    return index


def build_embedding_weights(word_index, embeddings_index):
    """Builds an embedding matrix for all words in vocab using embeddings_index
    """
    logger.info('Loading embeddings for all words in the corpus')
    embedding_dim = embeddings_index.values()[0].shape[-1]

    # +1 since tokenizer words are indexed from 1. 0 is reserved for padding and unknown words.
    embedding_weights = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        word_vector = embeddings_index.get(word)
        if word_vector is not None:
            # Words not in embedding will be all zeros which can stand for padded words.
            embedding_weights[i] = word_vector

    return embedding_weights


def get_embeddings_index(embedding_type='glove.42B.300d'):
    """Retrieves embeddings index from embedding name. Will automatically download and cache as needed.

    Args:
        embedding_type: The embedding type to load.

    Returns:
        The embeddings indexed by word.
    """

    embeddings_index = _EMBEDDINGS_CACHE.get(embedding_type)
    if embeddings_index is not None:
        return embeddings_index

    data_obj = _EMBEDDING_TYPES.get(embedding_type)
    if data_obj is None:
        raise ValueError("Embedding name should be one of '{}'".format(_EMBEDDING_TYPES.keys()))

    cache_dir = os.path.expanduser(os.path.join('~', '.keras-text'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    file_path = get_file(embedding_type, origin=data_obj['url'], extract=True,
                         cache_dir=cache_dir, cache_subdir='embeddings')
    file_path = os.path.join(os.path.dirname(file_path), data_obj['file'])

    embeddings_index = _build_embeddings_index(file_path)
    _EMBEDDINGS_CACHE[embedding_type] = embeddings_index
    return embeddings_index
