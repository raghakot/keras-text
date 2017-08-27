from __future__ import absolute_import

import logging
import numpy as np
from fractions import Fraction


logger = logging.getLogger(__name__)


def equal_distribution_folds(y, folds=2):
    """Creates `folds` number of indices that has roughly balanced multi-label distribution.

    Args:
        y: The multi-label outputs.
        folds: The number of folds to create.

    Returns:
        `folds` number of indices that have roughly equal multi-label distributions.
    """
    n, classes = y.shape

    # Compute sample distribution over classes
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()

    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(n):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)

    logger.debug("Fold distributions:")
    logger.debug(fold_dist)
    return index_list


def multi_label_train_test_split(y, test_size=0.2):
    """Creates a test split with roughly the same multi-label distribution in `y`.

    Args:
        y: The multi-label outputs.
        test_size: The test size in [0, 1]

    Returns:
        The train and test indices.
    """
    if test_size <= 0 or test_size >= 1:
        raise ValueError("`test_size` should be between 0 and 1")

    # Find the smallest rational number.
    frac = Fraction(test_size).limit_denominator()
    test_folds, total_folds = frac.numerator, frac.denominator
    logger.warn('Inferring test_size as {}/{}. Generating {} folds. The algorithm might fail if denominator is large.'
                .format(test_folds, total_folds, total_folds))

    folds = equal_distribution_folds(y, folds=total_folds)
    test_indices = np.concatenate(folds[:test_folds])
    train_indices = np.concatenate(folds[test_folds:])
    return train_indices, test_indices
