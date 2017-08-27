from __future__ import absolute_import

import numpy as np
import pickle
import joblib
import jsonpickle

from jsonpickle.ext import numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()


def dump(obj, file_name):
    if file_name.endswith('.json'):
        with open(file_name, 'w') as f:
            f.write(jsonpickle.dumps(obj))
        return

    if isinstance(obj, np.ndarray):
        np.save(file_name, obj)
        return

    # Using joblib instead of pickle because of http://bugs.python.org/issue11564
    joblib.dump(obj, file_name, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_name):
    if file_name.endswith('.json'):
        with open(file_name, 'r') as f:
            return jsonpickle.loads(f.read())

    if file_name.endswith('.npy'):
        return np.load(file_name)

    return joblib.load(file_name)
