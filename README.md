# Keras Text Classification Library
[![Build Status](https://travis-ci.org/raghakot/keras-text.svg?branch=master)](https://travis-ci.org/raghakot/keras-text)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/raghakot/keras-text/blob/master/LICENSE)
[![Slack](https://img.shields.io/badge/slack-discussion-E01563.svg)](https://join.slack.com/t/keras-text/shared_invite/MjMzNDU3NDAxODMxLTE1MDM4NTg0MTktNzgxZTNjM2E4Zg)

keras-text is a text classification library for training and developing new models. 

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
