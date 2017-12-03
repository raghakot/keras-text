from __future__ import absolute_import

import abc
import logging
import spacy

from . import utils
import numpy as np

from copy import deepcopy
from collections import defaultdict, OrderedDict
from multiprocessing import cpu_count

from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences
from keras.utils.generic_utils import Progbar


logger = logging.getLogger(__name__)


class _CountTracker(object):
    """Helper class to track counts of various document hierarchies in the corpus.
    For example, if the tokenizer can tokenize docs as (docs, paragraph, sentences, words), then this utility
    will track number of paragraphs, number of sentences within paragraphs and number of words within sentence.
    """

    def __init__(self):
        self._prev_indices = None
        self._local_counts = None
        self.counts = None

    def update(self, indices):
        """Updates counts based on indices. The algorithm tracks the index change at i and
        update global counts for all indices beyond i with local counts tracked so far.
        """
        # Initialize various lists for the first time based on length of indices.
        if self._prev_indices is None:
            self._prev_indices = indices

            # +1 to track token counts in the last index.
            self._local_counts = np.full(len(indices) + 1, 1)
            self._local_counts[-1] = 0
            self.counts = [[] for _ in range(len(self._local_counts))]

        has_reset = False
        for i in range(len(indices)):
            # index value changed. Push all local values beyond i to count and reset those local_counts.
            # For example, if document index changed, push counts on sentences and tokens and reset their local_counts
            # to indicate that we are tracking those for new document. We need to do this at all document hierarchies.
            if indices[i] > self._prev_indices[i]:
                self._local_counts[i] += 1
                has_reset = True
                for j in range(i + 1, len(self.counts)):
                    self.counts[j].append(self._local_counts[j])
                    self._local_counts[j] = 1

        # If none of the aux indices changed, update token count.
        if not has_reset:
            self._local_counts[-1] += 1
        self._prev_indices = indices[:]

    def finalize(self):
        """This will add the very last document to counts. We also get rid of counts[0] since that
        represents document level which doesnt come under anything else. We also convert all count
        values to numpy arrays so that stats can be computed easily.
        """
        for i in range(1, len(self._local_counts)):
            self.counts[i].append(self._local_counts[i])
        self.counts.pop(0)

        for i in range(len(self.counts)):
            self.counts[i] = np.array(self.counts[i])


def _apply_generator(texts, apply_fn):
    for text in texts:
        yield apply_fn(text)


def _append(lst, indices, value):
    """Adds `value` to `lst` list indexed by `indices`. Will create sub lists as required.
    """
    for i, idx in enumerate(indices):
        # We need to loop because sometimes indices can increment by more than 1 due to missing tokens.
        # Example: Sentence with no words after filtering words.
        while len(lst) <= idx:
            # Update max counts whenever a new sublist is created.
            # There is no need to worry about indices beyond `i` since they will end up creating new lists as well.
            lst.append([])
        lst = lst[idx]

    # Add token and update token max count.
    lst.append(value)


def _recursive_apply(lst, apply_fn):
    if len(lst) > 0 and not isinstance(lst[0], list):
        for i in range(len(lst)):
            lst[i] = apply_fn(lst[i])
    else:
        for sub_list in lst:
            _recursive_apply(sub_list, apply_fn)


def _to_unicode(text):
    if not isinstance(text, unicode):
        text = text.decode('utf-8')
    return text


def _parse_spacy_kwargs(**kwargs):
    """Supported args include:

    Args:
        n_threads/num_threads: Number of threads to use. Uses num_cpus - 1 by default.
        batch_size: The number of texts to accumulate into a common working set before processing.
            (Default value: 1000)
    """
    n_threads = kwargs.get('n_threads') or kwargs.get('num_threads')
    batch_size = kwargs.get('batch_size')

    if n_threads is None or n_threads is -1:
        n_threads = cpu_count() - 1
    if batch_size is None or batch_size is -1:
        batch_size = 1000
    return n_threads, batch_size


def _pad_token_sequences(sequences, max_tokens=None,
                         padding='pre', truncating='pre', value=0.):
    return keras_pad_sequences(sequences, maxlen=max_tokens, padding=padding, truncating=truncating, value=value)


def _pad_sent_sequences(sequences, max_sentences=None, max_tokens=None,
                        padding='pre', truncating='pre', value=0.):
    # Infer max lengths if needed.
    if max_sentences is None or max_tokens is None:
        max_sentences_computed = 0
        max_tokens_computed = 0
        for sent_seq in sequences:
            max_sentences_computed = max(max_sentences_computed, len(sent_seq))
            max_tokens_computed = max(max_tokens_computed, np.max([len(token_seq) for token_seq in sent_seq]))

        # Only use inferred values for None.
        max_sentences = min(max_sentences, max_sentences_computed)
        max_tokens = min(max_tokens, max_tokens_computed)

    result = np.ones(shape=(len(sequences), max_sentences, max_tokens)) * value

    for idx, sent_seq in enumerate(sequences):
        # empty list/array was found
        if not len(sent_seq):
            continue
        if truncating == 'pre':
            trunc = sent_seq[-max_sentences:]
        elif truncating == 'post':
            trunc = sent_seq[:max_sentences]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # Apply padding.
        if padding == 'post':
            result[idx, :len(trunc)] = _pad_token_sequences(trunc, max_tokens, padding, truncating, value)
        elif padding == 'pre':
            result[idx, -len(trunc):] = _pad_token_sequences(trunc, max_tokens, padding, truncating, value)
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return result


def pad_sequences(sequences, max_sentences=None, max_tokens=None,
                  padding='pre', truncating='post', value=0.):
    """Pads each sequence to the same length (length of the longest sequence or provided override).

    Args:
        sequences: list of list (samples, words) or list of list of list (samples, sentences, words)
        max_sentences: The max sentence length to use. If None, largest sentence length is used.
        max_tokens: The max word length to use. If None, largest word length is used.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than max_sentences or max_tokens
            either in the beginning or in the end of the sentence or word sequence respectively.
        value: The padding value.

    Returns:
        Numpy array of (samples, max_sentences, max_tokens) or (samples, max_tokens) depending on the sequence input.

    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`.
    """

    # Determine if input is (samples, max_sentences, max_tokens) or not.
    if isinstance(sequences[0][0], list):
        x = _pad_sent_sequences(sequences, max_sentences, max_tokens, padding, truncating, value)
    else:
        x = _pad_token_sequences(sequences, max_tokens, padding, truncating, value)
    return np.array(x, dtype='int32')


# def pad_sequences1(sequences, max_values, padding='pre', truncating='post', pad_value=0, dynamic_max=True):
#     computed_max_values = [0] * len(max_values)
#
#     def _compute_max(lst, level):
#         for lst in


def unicodify(texts):
    """Encodes all text sequences as unicode. This is a python2 hassle.

    Args:
        texts: The sequence of texts.

    Returns:
        Unicode encoded sequences.
    """
    return [_to_unicode(text) for text in texts]


class Tokenizer(object):

    def __init__(self,
                 lang='en',
                 lower=True):
        """Encodes text into `(samples, aux_indices..., token)` where each token is mapped to a unique index starting
        from `1`. Note that `0` is a reserved for unknown tokens.

        Args:
            lang: The spacy language to use. (Default value: 'en')
            lower: Lower cases the tokens if True. (Default value: True)
        """

        self.lang = lang
        self.lower = lower

        self._token2idx = dict()
        self._idx2token = dict()
        self._token_counts = defaultdict(int)

        self._num_texts = 0
        self._counts = None

    @abc.abstractmethod
    def token_generator(self, texts, **kwargs):
        """Generator for yielding tokens. You need to implement this method.

        Args:
            texts: list of text items to tokenize.
            **kwargs: The kwargs propagated from `build_vocab_and_encode` or `encode_texts` call.

        Returns:
            `(text_idx, aux_indices..., token)` where aux_indices are optional. For example, if you want to vectorize
                `texts` as `(text_idx, sentences, words), you should return `(text_idx, sentence_idx, word_token)`.
                Similarly, you can include paragraph, page level information etc., if needed.
        """
        raise NotImplementedError()

    def create_token_indices(self, tokens):
        """If `apply_encoding_options` is inadequate, one can retrieve tokens from `self.token_counts`, filter with
        a desired strategy and regenerate `token_index` using this method. The token index is subsequently used
        when `encode_texts` or `decode_texts` methods are called.
        """
        # Since 0 is reserved.
        indices = list(range(1, len(tokens) + 1))
        self._token2idx = dict(list(zip(tokens, indices)))
        self._idx2token = dict(list(zip(indices, tokens)))

    def apply_encoding_options(self, min_token_count=1, max_tokens=None):
        """Applies the given settings for subsequent calls to `encode_texts` and `decode_texts`. This allows you to
        play with different settings without having to re-run tokenization on the entire corpus.

        Args:
            min_token_count: The minimum token count (frequency) in order to include during encoding. All tokens
                below this frequency will be encoded to `0` which corresponds to unknown token. (Default value = 1)
            max_tokens: The maximum number of tokens to keep, based their frequency. Only the most common `max_tokens`
                tokens will be kept. Set to None to keep everything. (Default value: None)
        """
        if not self.has_vocab:
            raise ValueError("You need to build the vocabulary using `build_vocab` "
                             "before using `apply_encoding_options`")
        if min_token_count < 1:
            raise ValueError("`min_token_count` should atleast be 1")

        # Remove tokens with freq < min_token_count
        token_counts = list(self._token_counts.items())
        token_counts = filter(lambda x: x[1] >= min_token_count, token_counts)

        # Clip to max_tokens.
        if max_tokens is not None:
            token_counts.sort(key=lambda x: x[1], reverse=True)
            filtered_tokens = zip(*token_counts)[0]
            filtered_tokens = filtered_tokens[:max_tokens]
        else:
            filtered_tokens = zip(*token_counts)[0]

        # Generate indices based on filtered tokens.
        self.create_token_indices(filtered_tokens)

    def encode_texts(self, texts, include_oov=False, verbose=1, **kwargs):
        """Encodes the given texts using internal vocabulary with optionally applied encoding options. See
        ``apply_encoding_options` to set various options.

        Args:
            texts: The list of text items to encode.
            include_oov: True to map unknown (out of vocab) tokens to 0. False to exclude the token.
            verbose: The verbosity level for progress. Can be 0, 1, 2. (Default value = 1)
            **kwargs: The kwargs for `token_generator`.

        Returns:
            The encoded texts.
        """
        if not self.has_vocab:
            raise ValueError("You need to build the vocabulary using `build_vocab` before using `encode_texts`")

        progbar = Progbar(len(texts), verbose=verbose, interval=0.25)
        encoded_texts = []
        for token_data in self.token_generator(texts, **kwargs):
            indices, token = token_data[:-1], token_data[-1]

            token_idx = self._token2idx.get(token)
            if token_idx is None and include_oov:
                token_idx = 0

            if token_idx is not None:
                _append(encoded_texts, indices, token_idx)

            # Update progressbar per document level.
            progbar.update(indices[0])

        # All done. Finalize progressbar.
        progbar.update(len(texts), force=True)
        return encoded_texts

    def decode_texts(self, encoded_texts, unknown_token="<UNK>", inplace=True):
        """Decodes the texts using internal vocabulary. The list structure is maintained.

        Args:
            encoded_texts: The list of texts to decode.
            unknown_token: The placeholder value for unknown token. (Default value: "<UNK>")
            inplace: True to make changes inplace. (Default value: True)

        Returns:
            The decoded texts.
        """
        if len(self._token2idx) == 0:
            raise ValueError("You need to build vocabulary using `build_vocab` before using `decode_texts`")

        if not inplace:
            encoded_texts = deepcopy(encoded_texts)
        _recursive_apply(encoded_texts,
                         lambda token_id: self._idx2token.get(token_id) or unknown_token)
        return encoded_texts

    def build_vocab(self, texts, verbose=1, **kwargs):
        """Builds the internal vocabulary and computes various statistics.

        Args:
            texts: The list of text items to encode.
            verbose: The verbosity level for progress. Can be 0, 1, 2. (Default value = 1)
            **kwargs: The kwargs for `token_generator`.
        """
        if self.has_vocab:
            logger.warn("Tokenizer already has existing vocabulary. Overriding and building new vocabulary.")

        progbar = Progbar(len(texts), verbose=verbose, interval=0.25)
        count_tracker = _CountTracker()

        self._token_counts.clear()
        self._num_texts = len(texts)

        for token_data in self.token_generator(texts, **kwargs):
            indices, token = token_data[:-1], token_data[-1]
            count_tracker.update(indices)
            self._token_counts[token] += 1

            # Update progressbar per document level.
            progbar.update(indices[0])

        # Generate token2idx and idx2token.
        self.create_token_indices(self._token_counts.keys())

        # All done. Finalize progressbar update and count tracker.
        count_tracker.finalize()
        self._counts = count_tracker.counts
        progbar.update(len(texts), force=True)

    def get_counts(self, i):
        """Numpy array of count values for aux_indices. For example, if `token_generator` generates
        `(text_idx, sentence_idx, word)`, then `get_counts(0)` returns the numpy array of sentence lengths across
        texts. Similarly, `get_counts(1)` will return the numpy array of token lengths across sentences.

        This is useful to plot histogram or eyeball the distributions. For getting standard statistics, you can use
        `get_stats` method.
        """
        if not self.has_vocab:
            raise ValueError("You need to build the vocabulary using `build_vocab` before using `get_counts`")
        return self._counts[i]

    def get_stats(self, i):
        """Gets the standard statistics for aux_index `i`. For example, if `token_generator` generates
        `(text_idx, sentence_idx, word)`, then `get_stats(0)` will return various statistics about sentence lengths
        across texts. Similarly, `get_counts(1)` will return statistics of token lengths across sentences.

        This information can be used to pad or truncate inputs.
        """
        # OrderedDict to always show same order if printed.
        result = OrderedDict()
        result['min'] = np.min(self._counts[i])
        result['max'] = np.max(self._counts[i])
        result['std'] = np.std(self._counts[i])
        result['mean'] = np.mean(self._counts[i])
        return result

    def save(self, file_path):
        """Serializes this tokenizer to a file.

        Args:
            file_path: The file path to use.
        """
        utils.dump(self, file_path)

    @staticmethod
    def load(file_path):
        """Loads the Tokenizer from a file.

        Args:
            file_path: The file path to use.

        Returns:
            The `Dataset` instance.
        """
        return utils.load(file_path)

    @property
    def has_vocab(self):
        return len(self._token_counts) > 0 and self._counts is not None

    @property
    def token_index(self):
        """Dictionary of token -> idx mappings. This can change with calls to `apply_encoding_options`.
        """
        return self._token2idx

    @property
    def token_counts(self):
        """Dictionary of token -> count values for the text corpus used to `build_vocab`.
        """
        return self._token_counts

    @property
    def num_tokens(self):
        """Number of unique tokens for use in enccoding/decoding.
        This can change with calls to `apply_encoding_options`.
        """
        return len(self._token2idx)

    @property
    def num_texts(self):
        """The number of texts used to build the vocabulary.
        """
        return self._num_texts


class WordTokenizer(Tokenizer):

    def __init__(self,
                 lang='en',
                 lower=True,
                 lemmatize=False,
                 remove_punct=True,
                 remove_digits=True,
                 remove_stop_words=False,
                 exclude_oov=False,
                 exclude_pos_tags=None,
                 exclude_entities=['PERSON']):
        """Encodes text into `(samples, words)`

        Args:
            lang: The spacy language to use. (Default value: 'en')
            lower: Lower cases the tokens if True. (Default value: True)
            lemmatize: Lemmatizes words when set to True. This also makes the word lower case
                irrespective if the `lower` setting. (Default value: False)
            remove_punct: Removes punct words if True. (Default value: True)
            remove_digits: Removes digit words if True. (Default value: True)
            remove_stop_words: Removes stop words if True. (Default value: False)
            exclude_oov: Exclude words that are out of spacy embedding's vocabulary.
                By default, GloVe 1 million, 300 dim are used. You can override spacy vocabulary with a custom
                embedding to change this. (Default value: False)
            exclude_pos_tags: A list of parts of speech tags to exclude. Can be any of spacy.parts_of_speech.IDS
                (Default value: None)
            exclude_entities: A list of entity types to be excluded.
                Supported entity types can be found here: https://spacy.io/docs/usage/entity-recognition#entity-types
                (Default value: ['PERSON'])
        """

        super(WordTokenizer, self).__init__(lang, lower)
        self.lemmatize = lemmatize
        self.remove_punct = remove_punct
        self.remove_digits = remove_digits
        self.remove_stop_words = remove_stop_words

        self.exclude_oov = exclude_oov
        self.exclude_pos_tags = set(exclude_pos_tags or [])
        self.exclude_entities = set(exclude_entities or [])

    def _apply_options(self, token):
        """Applies various filtering and processing options on token.

        Returns:
            The processed token. None if filtered.
        """
        # Apply work token filtering.
        if token.is_punct and self.remove_punct:
            return None
        if token.is_stop and self.remove_stop_words:
            return None
        if token.is_digit and self.remove_digits:
            return None
        if token.is_oov and self.exclude_oov:
            return None
        if token.pos_ in self.exclude_pos_tags:
            return None
        if token.ent_type_ in self.exclude_entities:
            return None

        # Lemmatized ones are already lowered.
        if self.lemmatize:
            return token.lemma_
        if self.lower:
            return token.lower_
        return token.orth_

    def token_generator(self, texts, **kwargs):
        """Yields tokens from texts as `(text_idx, word)`

        Args:
            texts: The list of texts.
            **kwargs: Supported args include:
                n_threads/num_threads: Number of threads to use. Uses num_cpus - 1 by default.
                batch_size: The number of texts to accumulate into a common working set before processing.
                    (Default value: 1000)
        """
        # Perf optimization. Only process what is necessary.
        n_threads, batch_size = _parse_spacy_kwargs(**kwargs)
        nlp = spacy.load(self.lang)

        disabled = ['parser']
        if len(self.exclude_entities) > 0:
            disabled.append('ner')

        kwargs = {
            'batch_size': batch_size,
            'n_threads': n_threads,
            'disable': disabled
        }

        for text_idx, doc in enumerate(nlp.pipe(texts, **kwargs)):
            for word in doc:
                processed_word = self._apply_options(word)
                if processed_word is not None:
                    yield text_idx, processed_word


class SentenceWordTokenizer(WordTokenizer):

    def __init__(self,
                 lang='en',
                 lower=True,
                 lemmatize=False,
                 remove_punct=True,
                 remove_digits=True,
                 remove_stop_words=False,
                 exclude_oov=False,
                 exclude_pos_tags=None,
                 exclude_entities=['PERSON']):
        """Encodes text into `(samples, sentences, words)`

        Args:
            lang: The spacy language to use. (Default value: 'en')
            lower: Lower cases the tokens if True. (Default value: True)
            lemmatize: Lemmatizes words when set to True. This also makes the word lower case
                irrespective if the `lower` setting. (Default value: False)
            remove_punct: Removes punct words if True. (Default value: True)
            remove_digits: Removes digit words if True. (Default value: True)
            remove_stop_words: Removes stop words if True. (Default value: False)
            exclude_oov: Exclude words that are out of spacy embedding's vocabulary.
                By default, GloVe 1 million, 300 dim are used. You can override spacy vocabulary with a custom
                embedding to change this. (Default value: False)
            exclude_pos_tags: A list of parts of speech tags to exclude. Can be any of spacy.parts_of_speech.IDS
                (Default value: None)
            exclude_entities: A list of entity types to be excluded.
                Supported entity types can be found here: https://spacy.io/docs/usage/entity-recognition#entity-types
                (Default value: ['PERSON'])
        """
        super(SentenceWordTokenizer, self).__init__(lang,
                                                    lower,
                                                    lemmatize,
                                                    remove_punct,
                                                    remove_digits,
                                                    remove_stop_words,
                                                    exclude_oov,
                                                    exclude_pos_tags,
                                                    exclude_entities)

    def token_generator(self, texts, **kwargs):
        """Yields tokens from texts as `(text_idx, sent_idx, word)`

        Args:
            texts: The list of texts.
            **kwargs: Supported args include:
                n_threads/num_threads: Number of threads to use. Uses num_cpus - 1 by default.
                batch_size: The number of texts to accumulate into a common working set before processing.
                    (Default value: 1000)
        """
        # Perf optimization. Only process what is necessary.
        n_threads, batch_size = _parse_spacy_kwargs(**kwargs)
        nlp = spacy.load(self.lang)

        disabled = []
        if len(self.exclude_entities) > 0:
            disabled.append('ner')

        kwargs = {
            'batch_size': batch_size,
            'n_threads': n_threads,
            'disable': disabled
        }

        for text_idx, doc in enumerate(nlp.pipe(texts, **kwargs)):
            for sent_idx, sent in enumerate(doc.sents):
                for word in sent:
                    processed_word = self._apply_options(word)
                    if processed_word is not None:
                        yield text_idx, sent_idx, processed_word


class CharTokenizer(Tokenizer):

    def __init__(self,
                 lang='en',
                 lower=True,
                 charset=None):
        """Encodes text into `(samples, characters)`

        Args:
            lang: The spacy language to use. (Default value: 'en')
            lower: Lower cases the tokens if True. (Default value: True)
            charset: The character set to use. For example `charset = 'abc123'`. If None, all characters will be used.
                (Default value: None)
        """
        super(CharTokenizer, self).__init__(lang, lower)
        self.charset = charset

    def token_generator(self, texts, **kwargs):
        """Yields tokens from texts as `(text_idx, character)`
        """
        for text_idx, text in enumerate(texts):
            if self.lower:
                text = text.lower()
            for char in text:
                yield text_idx, char


class SentenceCharTokenizer(CharTokenizer):

    def __init__(self,
                 lang='en',
                 lower=True,
                 charset=None):
        """Encodes text into `(samples, sentences, characters)`

        Args:
            lang: The spacy language to use. (Default value: 'en')
            lower: Lower cases the tokens if True. (Default value: True)
            charset: The character set to use. For example `charset = 'abc123'`. If None, all characters will be used.
                (Default value: None)
        """
        super(SentenceCharTokenizer, self).__init__(lang, lower, charset)

    def token_generator(self, texts, **kwargs):
        """Yields tokens from texts as `(text_idx, sent_idx, character)`

        Args:
            texts: The list of texts.
            **kwargs: Supported args include:
                n_threads/num_threads: Number of threads to use. Uses num_cpus - 1 by default.
                batch_size: The number of texts to accumulate into a common working set before processing.
                    (Default value: 1000)
        """
        # Perf optimization. Only process what is necessary.
        n_threads, batch_size = _parse_spacy_kwargs(**kwargs)
        nlp = spacy.load(self.lang)

        kwargs = {
            'batch_size': batch_size,
            'n_threads': n_threads,
            'disable': ['ner']
        }

        # Perf optimization: Lower the entire text instead of individual tokens.
        texts_gen = _apply_generator(texts, lambda x: x.lower()) if self.lower else texts
        for text_idx, doc in enumerate(nlp.pipe(texts_gen, **kwargs)):
            for sent_idx, sent in enumerate(doc.sents):
                for word in sent:
                    for char in word:
                        yield text_idx, sent_idx, char


if __name__ == '__main__':
    texts = [
        "HELLO world hello. How are you today? Did you see the S.H.I.E.L.D?",
        "Quick brown fox. Ran over the, building 1234?",
    ]

    texts = unicodify(texts)
    tokenizer = SentenceWordTokenizer()
    tokenizer.build_vocab(texts)
    tokenizer.apply_encoding_options(max_tokens=5)
    encoded = tokenizer.encode_texts(texts)
    decoded = tokenizer.decode_texts(encoded, inplace=False)
    w = 1
