import sys
default_stdout = sys.stdout
default_stderr = sys.stderr
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = default_stdout
sys.stderr = default_stderr

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras_text.models import SentenceModelFactory, AttentionRNN
from keras_text.generators import ProcessingSequence
from keras_text.processing import pad_sequences
from keras_text.data import TextDataset
import pandas as pd

df = pd.read_json('datasets/yelp_academic_dataset_review.json', lines=True, encoding='utf-8')
ds = TextDataset.load('datasets/yelp_sent_dataset')
ds.tokenizer.apply_encoding_options(min_token_count=5)

max_sents = 50
max_words = 100
factory = SentenceModelFactory(len(ds.labels), ds.tokenizer.token_index,
                               max_sents, max_words,
                               embedding_type='glove.6B.100d')

model = factory.build_model(token_encoder_model=AttentionRNN(),
                            sentence_encoder_model=AttentionRNN())

model.compile(optimizer=Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

X = df['text'][ds.train_indices]
y = ds.y[ds.train_indices]

X_test = df['text'][ds.test_indices]
y_test = ds.y[ds.test_indices]


def process_fn(x):
    x = ds.tokenizer.encode_texts(x, verbose=0, n_threads=1)
    return pad_sequences(x, max_sentences=max_sents, max_tokens=max_words)


train_gen = ProcessingSequence(X, y, process_fn=process_fn, batch_size=128)
val_gen = ProcessingSequence(X_test, y_test, process_fn=process_fn, batch_size=128)

run_id = 1234
tb = TensorBoard(log_dir='../logs/{}'.format(run_id))
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('model.hdf5', monitor='val_loss', save_best_only=True)

model.fit_generator(train_gen, steps_per_epoch=len(train_gen), epochs=10,
                    workers=8, validation_data=val_gen, validation_steps=len(val_gen),
                    callbacks=[model_checkpoint, tb, early_stopping])
