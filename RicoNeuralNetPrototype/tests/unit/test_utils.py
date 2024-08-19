from RicoNeuralNetPrototype.utils.input_data import (
    generate_spiral_data,
    generate_gaussian_mixtures,
    generate_circles_within_circles,
    visualize_2D_data,
    partition_data
)

skip_visualize = True

def test_input_data_utils():
    # Create a scatter plot
    X, y = generate_spiral_data(n_points=200, classes=5)
    visualize_2D_data(X, y, "Spiral Data Visualization", skip_visualize)

    X, y = generate_gaussian_mixtures(200, classes=5)
    visualize_2D_data(X,y,"Gaussian Mixture Visualization", skip_visualize)

    X_train, y_train, X_holdout, y_holdout, X_test, y_test = partition_data(X=X, y=y)
    assert(X_train.shape[0] == 140) 
    assert(y_train.shape[0] == 140) 
    assert(X_holdout.shape[0] == 40) 
    assert(y_holdout.shape[0] == 40) 
    assert(X_test.shape[0] == 20) 
    assert(y_test.shape[0] == 20) 

    X, y = generate_circles_within_circles(n_points=200, classes=3)
    visualize_2D_data(X,y,"Circles Within Circles Visualization", skip_visualize)

