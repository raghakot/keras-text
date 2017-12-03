# Keras Text Classification Library
[![Build Status](https://travis-ci.org/raghakot/keras-text.svg?branch=master)](https://travis-ci.org/raghakot/keras-text)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/raghakot/keras-text/blob/master/LICENSE)
[![Slack](https://img.shields.io/badge/slack-discussion-E01563.svg)](https://join.slack.com/t/keras-text/shared_invite/MjMzNDU3NDAxODMxLTE1MDM4NTg0MTktNzgxZTNjM2E4Zg)

keras-text is a one-stop text classification library implementing various state of the art models with a clean and 
extendable interface to implement custom architectures.

## Quick start

### Create a tokenizer to build your vocabulary

- To represent you dataset as `(docs, words)` use `WordTokenizer`
- To represent you dataset as `(docs, sentences, words)` use `SentenceWordTokenizer`
- To create arbitrary hierarchies, extend `Tokenizer` and implement the `token_generator` method.

```python
from keras_text.processing import WordTokenizer


tokenizer = WordTokenizer()
tokenizer.build_vocab(texts)
```

Want to tokenize with character tokens to leverage character models? Use `CharTokenizer`.


### Build a dataset

A dataset encapsulates tokenizer, X, y and the test set. This allows you to focus your efforts on 
trying various architectures/hyperparameters without having to worry about inconsistent evaluation. A dataset can be 
saved and loaded from the disk.

```python
from keras_text.data import Dataset


ds = Dataset(X, y, tokenizer=tokenizer)
ds.update_test_indices(test_size=0.1)
ds.save('dataset')
```

The `update_test_indices` method automatically stratifies multi-class or multi-label data correctly.

### Build text classification models

See tests/ folder for usage.

#### Word based models

When dataset represented as `(docs, words)` word based models can be created using `TokenModelFactory`.

```python
from keras_text.models import TokenModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN


# RNN models can use `max_tokens=None` to indicate variable length words per mini-batch.
factory = TokenModelFactory(1, tokenizer.token_index, max_tokens=100, embedding_type='glove.6B.100d')
word_encoder_model = YoonKimCNN()
model = factory.build_model(token_encoder_model=word_encoder_model)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
``` 

Currently supported models include:

- [Yoon Kim CNN](https://arxiv.org/abs/1408.5882)
- Stacked RNNs
- Attention (with/without context) based RNN encoders.

`TokenModelFactory.build_model` uses the provided word encoder which is then classified via `Dense` block.

#### Sentence based models

When dataset represented as `(docs, sentences, words)` sentence based models can be created using `SentenceModelFactory`.

```python
from keras_text.models import SentenceModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN, AveragingEncoder


# Pad max sentences per doc to 500 and max words per sentence to 200.
# Can also use `max_sents=None` to allow variable sized max_sents per mini-batch.
factory = SentenceModelFactory(10, tokenizer.token_index, max_sents=500, max_tokens=200, embedding_type='glove.6B.100d')
word_encoder_model = AttentionRNN()
sentence_encoder_model = AttentionRNN()

# Allows you to compose arbitrary word encoders followed by sentence encoder.
model = factory.build_model(word_encoder_model, sentence_encoder_model)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
``` 

Currently supported models include:

- [Yoon Kim CNN](https://arxiv.org/abs/1408.5882)
- Stacked RNNs
- Attention (with/without context) based RNN encoders.

`SentenceModelFactory.build_model` created a tiered model where words within a sentence is first encoded using  
`word_encoder_model`. All such encodings per sentence is then encoded using `sentence_encoder_model`.

- [Hierarchical attention networks](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf) 
(HANs) can be build by composing two attention based RNN models. This is useful when a document is very large.
- For smaller document a reasonable way to encode sentences is to average words within it. This can be done by using
`token_encoder_model=AveragingEncoder()`
- Mix and match encoders as you see fit for your problem.


## Resources

TODO: Update documentation and add notebook examples.

Stay tuned for better documentation and examples. 
Until then, the best resource is to refer to the [API docs](https://raghakot.github.io/keras-text/)


## Installation

1) Install [keras](https://github.com/fchollet/keras/blob/master/README.md#installation) 
with theano or tensorflow backend. Note that this library requires Keras > 2.0

2) Install keras-text
> From sources
```bash
sudo python setup.py install
```

> PyPI package
```bash
sudo pip install keras-text
```

3) Download target spacy model

keras-text uses the excellent spacy library for tokenization. See instructions on how to 
[download model](https://spacy.io/docs/usage/models#download) for target language.


## Citation

Please cite keras-text in your publications if it helped your research. Here is an example BibTeX entry:

```
@misc{raghakotkerastext
  title={keras-text},
  author={Kotikalapudi, Raghavendra and contributors},
  year={2017},
  publisher={GitHub},
  howpublished={\url{https://github.com/raghakot/keras-text}},
}
```
