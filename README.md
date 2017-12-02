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

TODO: Update documentation and add notebook examples.


## Resources

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
