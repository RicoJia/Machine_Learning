import torch

from RicoNeuralNetPrototype.utils.input_data import (
    generate_circles_within_circles, generate_gaussian_mixtures,
    generate_spiral_data, generate_xor_data, partition_data, to_tensor,
    visualize_2D_data)

skip_visualize = True

def test_input_data_utils():
    # Create a scatter plot
    X, y = generate_spiral_data(n_points=200, classes=5)
    visualize_2D_data(X, y, "Spiral Data Visualization", skip_visualize)

    X, y = generate_gaussian_mixtures(200, classes=5)
    visualize_2D_data(X,y,"Gaussian Mixture Visualization", skip_visualize)

    X_train, y_train, X_test, y_test, X_validation, y_validation = partition_data(X=X, y=y)
    assert(X_train.shape[0] == 140) 
    assert(y_train.shape[0] == 140) 
    assert(X_test.shape[0] == 40) 
    assert(y_test.shape[0] == 40) 
    assert(X_validation.shape[0] == 20) 
    assert(y_validation.shape[0] == 20) 

    X, y = generate_circles_within_circles(n_points=200, classes=3)
    visualize_2D_data(X,y,"Circles Within Circles Visualization", skip_visualize)

    X, y = generate_xor_data(n_points=200)
    visualize_2D_data(X,y,"Xor Data", skip_visualize)
    
    X_train_torch, y_train_torch =  to_tensor(X_train, y_train)
    assert(type(X_train_torch) == torch.Tensor)