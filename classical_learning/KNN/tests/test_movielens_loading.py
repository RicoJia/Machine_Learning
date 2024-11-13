import os

import numpy as np
from src import load_movielens_data


def test_load_movielens_data():
    data = load_movielens_data(os.path.join("data", "ml-100k"))
    assert data.shape == (943, 1682) and np.count_nonzero(data) == 100000
