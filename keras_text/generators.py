from __future__ import absolute_import

import numpy as np
from keras.utils import Sequence


class ProcessingSequence(Sequence):
    def __init__(self, X, y, batch_size, process_fn=None):
        """A `Sequence` implementation that can pre-process a mini-batch via `process_fn`

        Args:
            X: The numpy array of inputs.
            y: The numpy array of targets.
            batch_size: The generator mini-batch size.
            process_fn: The preprocessing function to apply on `X`
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.process_fn = process_fn or (lambda x: x)

    def __len__(self):
        return len(self.X) // self.batch_size

    def on_epoch_end(self):
        pass

    def __getitem__(self, batch_idx):
        batch_X = self.X[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        batch_y = self.y[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return self.process_fn(batch_X), batch_y


class BalancedSequence(Sequence):
    def __init__(self, X, y, batch_size, process_fn=None):
        """A `Sequence` implementation that returns balanced `y` by undersampling majority class.

        Args:
            X: The numpy array of inputs.
            y: The numpy array of targets.
            batch_size: The generator mini-batch size.
            process_fn: The preprocessing function to apply on `X`
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.process_fn = process_fn or (lambda x: x)

        self.pos_indices = np.where(y == 1)[0]
        self.neg_indices = np.where(y == 0)[0]
        self.n = min(len(self.pos_indices), len(self.neg_indices))
        self._index_array = None

    def __len__(self):
        # Reset batch after we are done with minority class.
        return (self.n * 2) // self.batch_size

    def on_epoch_end(self):
        # Reset batch after all minority indices are covered.
        self._index_array = None

    def __getitem__(self, batch_idx):
        if self._index_array is None:
            pos_indices = self.pos_indices.copy()
            neg_indices = self.neg_indices.copy()
            np.random.shuffle(pos_indices)
            np.random.shuffle(neg_indices)
            self._index_array = np.concatenate((pos_indices[:self.n], neg_indices[:self.n]))
            np.random.shuffle(self._index_array)

        indices = self._index_array[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        return self.process_fn(self.X[indices]), self.y[indices]
