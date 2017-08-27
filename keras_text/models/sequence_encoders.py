from __future__ import absolute_import

from keras.layers import Conv1D, Bidirectional, GRU
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.layers.merge import concatenate
from .layers import AttentionLayer, ConsumeMask


class SequenceEncoderBase(object):

    def __init__(self, dropout_rate=0.5):
        """Creates a new instance of sequence encoder.

        Args:
            dropout_rate: The final encoded output dropout.
        """
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        """Build the actual model here.

        Args:
            x: The encoded or embedded input sequence.

        Returns:
            The model output tensor.
        """
        x = self.build_model(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        return x

    def build_model(self, x):
        """Build your model graph here.

        Args:
            x: The encoded or embedded input sequence.

        Returns:
            The model output tensor without the classification block.
        """
        raise NotImplementedError()

    def requires_padding(self):
        """Return a boolean indicating whether this model expects inputs to be padded or not.
        """
        raise NotImplementedError()


class YoonKimCNN(SequenceEncoderBase):

    def __init__(self, num_filters=64, filter_sizes=[3, 4, 5], dropout_rate=0.5, **conv_kwargs):
        """Yoon Kim's shallow cnn model: https://arxiv.org/pdf/1408.5882.pdf

        Args:
            num_filters: The number of filters to use per `filter_size`. (Default value = 64)
            filter_sizes: The filter sizes for each convolutional layer. (Default value = [3, 4, 5])
            **cnn_kwargs: Additional args for building the `Conv1D` layer.
        """
        super(YoonKimCNN, self).__init__(dropout_rate)
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.conv_kwargs = conv_kwargs

    def build_model(self, x):
        pooled_tensors = []
        for filter_size in self.filter_sizes:
            x_i = Conv1D(self.num_filters, filter_size, activation='elu', **self.conv_kwargs)(x)
            x_i = GlobalMaxPooling1D()(x_i)
            pooled_tensors.append(x_i)

        x = pooled_tensors[0] if len(self.filter_sizes) == 1 else concatenate(pooled_tensors, axis=-1)
        return x

    def requires_padding(self):
        return True


class StackedRNN(SequenceEncoderBase):

    def __init__(self, rnn_class=GRU, hidden_dims=[50, 50], bidirectional=True, dropout_rate=0.5, **rnn_kwargs):
        """Creates a stacked RNN.

        Args:
            rnn_class: The type of RNN to use.
            hidden_dims: The hidden dims for corresponding stacks of RNNs.
            bidirectional: Whether to use bidirectional encoding.
            **rnn_kwargs: Additional args for building the RNN.
        """
        super(StackedRNN, self).__init__(dropout_rate)
        self.rnn_class = rnn_class
        self.hidden_dims = hidden_dims
        self.bidirectional = bidirectional
        self.rnn_kwargs = rnn_kwargs

    def build_model(self, x):
        for i, n in enumerate(self.hidden_dims):
            is_last_layer = i == len(self.hidden_dims) - 1
            rnn = self.rnn_class(n, return_sequences=not is_last_layer, **self.rnn_kwargs)
            if self.bidirectional:
                x = Bidirectional(rnn)(x)
            else:
                x = rnn(x)
        return x

    def requires_padding(self):
        return False


class AttentionRNN(SequenceEncoderBase):

    def __init__(self, rnn_class=GRU, encoder_dims=50, bidirectional=True, dropout_rate=0.5, **rnn_kwargs):
        """Creates an RNN model with attention. The attention mechanism is implemented as described
        in https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf, but without
        sentence level attention.

        Args:
            rnn_class: The type of RNN to use.
            encoder_dims: The number of hidden units of RNN.
            bidirectional: Whether to use bidirectional encoding.
            **rnn_kwargs: Additional args for building the RNN.
        """
        super(AttentionRNN, self).__init__(dropout_rate)
        self.rnn_class = rnn_class
        self.encoder_dims = encoder_dims
        self.bidirectional = bidirectional
        self.rnn_kwargs = rnn_kwargs

    def build_model(self, x):
        rnn = self.rnn_class(self.encoder_dims, return_sequences=True, **self.rnn_kwargs)
        if self.bidirectional:
            word_activations = Bidirectional(rnn)(x)
        else:
            word_activations = rnn(x)

        attention_layer = AttentionLayer()
        doc_vector = attention_layer(word_activations)
        self.attention_tensor = attention_layer.get_attention_tensor()
        return doc_vector

    def get_attention_tensor(self):
        if not hasattr(self, 'attention_tensor'):
            raise ValueError('You need to build the model first')
        return self.attention_tensor

    def requires_padding(self):
        return False


class AveragingEncoder(SequenceEncoderBase):

    def __init__(self, dropout_rate=0):
        """An encoder that averages sequence inputs.
        """
        super(AveragingEncoder, self).__init__(dropout_rate)

    def build_model(self, x):
        x = ConsumeMask()(x)
        x = GlobalAveragePooling1D()(x)
        return x

    def requires_padding(self):
        return False
