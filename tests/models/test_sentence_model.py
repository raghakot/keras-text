import pytest
from keras_text.models import SentenceModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN, AveragingEncoder


def _test_build(token_encoder_model, sentence_encoder_model):
    test_index = {'hello': 1, 'kitty': 2}

    if sentence_encoder_model.allows_dynamic_length():
        factory = SentenceModelFactory(10, test_index, max_sents=None, max_tokens=200, embedding_type=None)
        model = factory.build_model(token_encoder_model, sentence_encoder_model)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.summary()
    else:
        # Should fail since this model does not allow dynamic mini-batches.
        factory = SentenceModelFactory(10, test_index, max_sents=None, max_tokens=200, embedding_type=None)
        with pytest.raises(ValueError):
            factory.build_model(token_encoder_model, sentence_encoder_model)

        factory = SentenceModelFactory(10, test_index, max_sents=500, max_tokens=200, embedding_type=None)
        model = factory.build_model(token_encoder_model, sentence_encoder_model)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.summary()


def test_hierarchical_attention_model():
    _test_build(AttentionRNN(), AttentionRNN())


def test_combinations():
    encoders = [YoonKimCNN(), AttentionRNN(), StackedRNN(), AveragingEncoder()]
    for word_encoder in encoders:
        for sentence_encoder in encoders:
            print('Testing combination {}, {}'.format(word_encoder.__class__, sentence_encoder.__class__))
            _test_build(word_encoder, sentence_encoder)


if __name__ == '__main__':
    pytest.main([__file__])
