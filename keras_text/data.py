from __future__ import absolute_import

import logging
import numpy as np

from .import utils
from .import sampling

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit


logger = logging.getLogger(__name__)


class Dataset(object):

    def __init__(self, inputs, labels, test_indices=None, **kwargs):
        """Encapsulates all pieces of data to run an experiment. This is basically a bag of items that makes it
        easy to serialize and deserialize everything as a unit.

        Args:
            inputs: The raw model inputs. This can be set to None if you dont want
                to serialize this value when you save the dataset.
            labels: The raw output labels.
            test_indices: The optional test indices to use. Ideally, this should be generated one time and reused
                across experiments to make results comparable. `generate_test_indices` can be used generate first
                time indices.
            **kwargs: Additional key value items to store.
        """
        self.X = np.array(inputs)
        self.y = np.array(labels)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._test_indices = None
        self._train_indices = None
        self.test_indices = test_indices

        self.is_multi_label = isinstance(labels[0], (set, list, tuple))
        self.label_encoder = MultiLabelBinarizer() if self.is_multi_label else LabelBinarizer()
        self.y = self.label_encoder.fit_transform(self.y).flatten()

    def update_test_indices(self, test_size=0.1):
        """Updates `test_indices` property with indices of `test_size` proportion.

        Args:
            test_size: The test proportion in [0, 1] (Default value: 0.1)
        """
        if self.is_multi_label:
            self._train_indices, self._test_indices = sampling.multi_label_train_test_split(self.y, test_size)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
            self._train_indices, self._test_indices = next(sss.split(self.X, self.y))

    def save(self, file_path):
        """Serializes this dataset to a file.

        Args:
            file_path: The file path to use.
        """
        utils.dump(self, file_path)

    def train_val_split(self, split_ratio=0.1):
        """Generates train and validation sets from the training indices.

        Args:
            split_ratio: The split proportion in [0, 1] (Default value: 0.1)

        Returns:
            The stratified train and val subsets. Multi-label outputs are handled as well.
        """
        if self.is_multi_label:
            train_indices, val_indices = sampling.multi_label_train_test_split(self.y, split_ratio)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio)
            train_indices, val_indices = next(sss.split(self.X, self.y))
        return self.X[train_indices], self.X[val_indices], self.y[train_indices], self.y[val_indices]

    @staticmethod
    def load(file_path):
        """Loads the dataset from a file.

        Args:
            file_path: The file path to use.

        Returns:
            The `Dataset` instance.
        """
        return utils.load(file_path)

    @property
    def test_indices(self):
        return self._test_indices

    @test_indices.setter
    def test_indices(self, test_indices):
        if test_indices is None:
            self._train_indices = np.arange(0, len(self.y))
        else:
            self._test_indices = test_indices
            self._train_indices = np.setdiff1d(np.arange(0, len(self.y)), self.test_indices)

    @property
    def train_indices(self):
        return self._train_indices

    @property
    def labels(self):
        return self.label_encoder.classes_

    @property
    def num_classes(self):
        if len(self.y.shape) == 1:
            return 1
        else:
            return len(self.labels)
