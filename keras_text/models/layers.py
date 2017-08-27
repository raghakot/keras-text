from __future__ import absolute_import

import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints


def _softmax(x, dim):
    """Computes softmax along a specified dim. Keras currently lacks this feature.
    """

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.nn.softmax(x, dim)
    elif K.backend() is 'cntk':
        import cntk
        return cntk.softmax(x, dim)
    elif K.backend() == 'theano':
        # Theano cannot softmax along an arbitrary dim.
        # So, we will shuffle `dim` to -1 and un-shuffle after softmax.
        perm = np.arange(K.ndim(x))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_perm = K.permute_dimensions(x, perm)
        output = K.softmax(x_perm)

        # Permute back
        perm[dim], perm[-1] = perm[-1], perm[dim]
        output = K.permute_dimensions(x, output)
        return output
    else:
        raise ValueError("Backend '{}' not supported".format(K.backend()))


class AttentionLayer(Layer):
    """Attention layer that computes a learned attention over input sequence.

    For details, see papers:
    - https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    - http://colinraffel.com/publications/iclr2016feed.pdf (fig 1)

    Input:
        x: Input tensor of shape `(..., time_steps, features)` where `features` must be static (known).

    Output:
        2D tensor of shape `(..., features)`. i.e., `time_steps` axis is attended over and reduced.
    """

    def __init__(self,
                 kernel_initializer='he_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 use_context=True,
                 context_initializer='he_normal',
                 context_regularizer=None,
                 context_constraint=None,
                 attention_dims=None,
                 **kwargs):
        """
        Args:
            attention_dims: The dimensionality of the inner attention calculating neural network.
                For input `(32, 10, 300)`, with `attention_dims` of 100, the output is `(32, 10, 100)`.
                i.e., the attended words are 100 dimensional. This is then collapsed via summation to
                `(32, 10, 1)` to indicate the attention weights for 10 words.
                If set to None, `features` dims are used as `attention_dims`. (Default value: None)
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(AttentionLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_context = use_context
        self.context_initializer = initializers.get(context_initializer)
        self.context_regularizer = regularizers.get(context_regularizer)
        self.context_constraint = constraints.get(context_constraint)

        self.attention_dims = attention_dims
        self.supports_masking = True

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError("Expected input shape of `(..., time_steps, features)`, found `{}`".format(input_shape))

        attention_dims = input_shape[-1] if self.attention_dims is None else self.attention_dims
        self.kernel = self.add_weight(shape=(input_shape[-1], attention_dims),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(attention_dims, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.use_context:
            self.context_kernel = self.add_weight(shape=(attention_dims, ),
                                                  initializer=self.context_initializer,
                                                  name='context_kernel',
                                                  regularizer=self.context_regularizer,
                                                  constraint=self.context_constraint)
        else:
            self.context_kernel = None

        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        # x: [..., time_steps, features]
        # ut = [..., time_steps, attention_dims]
        ut = K.dot(x, self.kernel)
        if self.use_bias:
            ut = K.bias_add(ut, self.bias)

        ut = K.tanh(ut)
        if self.use_context:
            ut = ut * self.context_kernel

        # Collapse `attention_dims` to 1. This indicates the weight for each time_step.
        ut = K.sum(ut, axis=-1, keepdims=True)

        # Convert those weights into a distribution but along time axis.
        # i.e., sum of alphas along `time_steps` axis should be 1.
        self.at = _softmax(ut, dim=1)
        if mask is not None:
            self.at *= K.cast(K.expand_dims(mask, -1), K.floatx())

        # Weighted sum along `time_steps` axis.
        return K.sum(x * self.at, axis=-2)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_attention_tensor(self):
        if not hasattr(self, 'at'):
            raise ValueError('Attention tensor is available after calling this layer with an input')
        return self.at

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'context_initializer': initializers.serialize(self.context_initializer),
            'context_regularizer': regularizers.serialize(self.context_regularizer),
            'context_constraint': constraints.serialize(self.context_constraint)
        }
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConsumeMask(Layer):
    """Layer that prevents mask propagation.
    """

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x
