import pytest
from keras_text.models import TokenModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN


def _test_build(token_encoder_model):
    test_index = {'hello': 1, 'kitty': 2}

    if token_encoder_model.allows_dynamic_length():
        factory = TokenModelFactory(1, test_index, max_tokens=None, embedding_type=None)
        model = factory.build_model(token_encoder_model)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.summary()
    else:
        # Should fail since this model does not allow dynamic mini-batches.
        factory = TokenModelFactory(1, test_index, max_tokens=None, embedding_type=None)
        with pytest.raises(ValueError):
            factory.build_model(token_encoder_model)

        factory = TokenModelFactory(1, test_index, max_tokens=100, embedding_type=None)
        model = factory.build_model(token_encoder_model)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.summary()


def test_yoon_kim_cnn():
    _test_build(YoonKimCNN())


def test_attention_rnn():
    _test_build(AttentionRNN())


def test_stacked_rnn():
    _test_build(StackedRNN())


if __name__ == '__main__':
    pytest.main([__file__])
