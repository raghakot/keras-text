from __future__ import absolute_import

from keras.layers import Input, Embedding, Dense
from keras.models import Model

from ..embeddings import get_embeddings_index, build_embedding_weights
from .sequence_encoders import SequenceEncoderBase


class TokenModelFactory(object):
    def __init__(self, num_classes, token_index, max_tokens,
                 embedding_type='glove.6B.100d', embedding_dims=100):
        """Creates a `TokenModelFactory` instance for building various models that operate over
        (samples, max_tokens) input. The token can be character, word or any other elementary token.

        Args:
            num_classes: The number of output classes.
            token_index: The dictionary of token and its corresponding integer index value.
            max_tokens: The max number of tokens across all documents. This can be set to None for models that
                allow different word lengths per mini-batch.
            embedding_type: The embedding type to use. Set to None to use random embeddings.
                (Default value: 'glove.6B.100d')
            embedding_dims: The number of embedding dims to use for representing a word. This argument will be ignored
                when `embedding_type` is set. (Default value: 100)
        """
        self.num_classes = num_classes
        self.token_index = token_index
        self.max_tokens = max_tokens

        if embedding_type is not None:
            self.embeddings_index = get_embeddings_index(embedding_type)
            self.embedding_dims = self.embeddings_index.values()[0].shape[-1]
        else:
            self.embeddings_index = None
            self.embedding_dims = embedding_dims

    def build_model(self, token_encoder_model, trainable_embeddings=True, output_activation='softmax'):
        """Builds a model using the given `text_model`

        Args:
            token_encoder_model: An instance of `SequenceEncoderBase` for encoding all the tokens within a document.
                This encoding is then fed into a final `Dense` layer for classification.
            trainable_embeddings: Whether or not to fine tune embeddings.
            output_activation: The output activation to use. (Default value: 'softmax')
                Use:
                - `softmax` for binary or multi-class.
                - `sigmoid` for multi-label classification.
                - `linear` for regression output.

        Returns:
            The model output tensor.
        """
        if not isinstance(token_encoder_model, SequenceEncoderBase):
            raise ValueError("`token_encoder_model` should be an instance of `{}`".format(SequenceEncoderBase))

        if not token_encoder_model.allows_dynamic_length() and self.max_tokens is None:
            raise ValueError("The provided `token_encoder_model` does not allow variable length mini-batches. "
                             "You need to provide `max_tokens`")

        if self.embeddings_index is None:
            # The +1 is for unknown token index 0.
            embedding_layer = Embedding(len(self.token_index) + 1,
                                        self.embedding_dims,
                                        input_length=self.max_tokens,
                                        mask_zero=True,
                                        trainable=trainable_embeddings)
        else:
            embedding_layer = Embedding(len(self.token_index) + 1,
                                        self.embedding_dims,
                                        weights=[build_embedding_weights(self.token_index, self.embeddings_index)],
                                        input_length=self.max_tokens,
                                        mask_zero=True,
                                        trainable=trainable_embeddings)

        sequence_input = Input(shape=(self.max_tokens,), dtype='int32')
        x = embedding_layer(sequence_input)
        x = token_encoder_model(x)
        x = Dense(self.num_classes, activation=output_activation)(x)
        return Model(sequence_input, x)
