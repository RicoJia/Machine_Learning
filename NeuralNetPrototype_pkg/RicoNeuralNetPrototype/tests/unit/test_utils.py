import numpy as np
import torch

from RicoNeuralNetPrototype.utils.input_data import (
    generate_circles_within_circles,
    generate_gaussian_mixtures,
    generate_spiral_data,
    generate_xor_data,
    partition_data,
    to_one_hot,
    to_tensor,
    visualize_2D_data,
)

skip_visualize = True


def check_one_hot(y, one_hot, classes):
    assert y.ndim == 1
    assert one_hot.shape == (y.shape[0], classes)
    np.testing.assert_allclose(np.argmax(one_hot, axis=1), y)


def test_input_data_utils():
    # Create a scatter plot
    CLASS_NUM = 5
    X, y = generate_spiral_data(n_points=200, classes=CLASS_NUM)
    visualize_2D_data(X, y, "Spiral Data Visualization", skip_visualize)
    one_hot = to_one_hot(y)
    check_one_hot(y=y, one_hot=one_hot, classes=CLASS_NUM)

    X, y = generate_gaussian_mixtures(200, classes=5)
    visualize_2D_data(X, y, "Gaussian Mixture Visualization", skip_visualize)
    one_hot = to_one_hot(y)
    check_one_hot(y=y, one_hot=one_hot, classes=CLASS_NUM)

    X_train, y_train, X_test, y_test, X_validation, y_validation = partition_data(
        X=X, y=y
    )
    assert X_train.shape[0] == 140
    assert y_train.shape[0] == 140
    assert X_test.shape[0] == 40
    assert y_test.shape[0] == 40
    assert X_validation.shape[0] == 20
    assert y_validation.shape[0] == 20

    X, y = generate_circles_within_circles(n_points=200, classes=CLASS_NUM)
    visualize_2D_data(X, y, "Circles Within Circles Visualization", skip_visualize)
    one_hot = to_one_hot(y)
    check_one_hot(y=y, one_hot=one_hot, classes=CLASS_NUM)

    X, y = generate_xor_data(n_points=200)
    visualize_2D_data(X, y, "Xor Data", skip_visualize)
    one_hot = to_one_hot(y)
    check_one_hot(y=y, one_hot=one_hot, classes=2)

    # TODO Remember to remove
    print(f"Rico: {one_hot}")

    X_train_torch, y_train_torch = to_tensor(X_train, y_train)
    assert type(X_train_torch) == torch.Tensor
