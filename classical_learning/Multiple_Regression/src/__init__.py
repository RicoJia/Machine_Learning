# Fix for determinism.
import numpy as np

from .generate_regression_data import generate_regression_data
from .load_json_data import load_json_data
from .metrics import mean_squared_error
from .perceptron import Perceptron, transform_data
from .polynomial_regression import PolynomialRegression

np.random.seed(0)
