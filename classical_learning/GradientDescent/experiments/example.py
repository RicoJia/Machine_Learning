from your_code import GradientDescent, load_data, ZeroOneLoss, MultiClassGradientDescent, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# print('Starting example experiment')
#
# train_features, test_features, train_targets, test_targets = \
#     load_data('blobs', fraction=0.6)
# learner = GradientDescent('squared')
# learner.fit(train_features, train_targets)
# predictions = learner.predict(test_features)
#
# print('Finished example experiment')



def test_hinge_loss_backward():
    """
    Tests the backward pass of the hinge loss function
    """
    from your_code import HingeLoss
    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    w = np.array([1, 2, 3])
    y = np.array([1, -1])

    loss = HingeLoss(regularization=None)

    _true = np.array([-1.5, 2, 0.5])
    _est = loss.backward(X, w, y)
    print(_est)




def test_squared_loss_forward():
    """
    Tests the forward pass of the squared loss function
    """
    from your_code import SquaredLoss
    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    w = np.array([1, 2, 3])
    y = np.array([1, -1])

    loss = SquaredLoss(regularization=None)

    _true = 26.5
    _est = loss.forward(X, w, y)
    print(_est)



def make_predictions(features, targets, loss, regularization):
    """
    Fit and predict on the training set using gradient descent and default
    parameter values. Note that in practice, the testing set should be used for
    predictions. This code is just to common-sense check that your gradient
    descent algorithm can classify the data it was trained on.
    """
    from your_code import GradientDescent

    np.random.seed(0)
    learner = GradientDescent(loss=loss, regularization=regularization,
                              learning_rate=0.01, reg_param=0.05)
    learner.fit(features, targets, batch_size=None, max_iter=1000)

    print("actual targets: ", targets)
    return learner.predict(features)


def test_gradient_descent_blobs():
    """
    Tests the ability of the gradient descent algorithm to classify a linearly
    separable dataset.
    """
    features, _, targets, _ = load_data('blobs')

    hinge = make_predictions(features, targets, 'hinge', None)
    # assert np.all(hinge == targets)

    # l1_hinge = make_predictions(features, targets, 'hinge', 'l1')
    # # assert np.all(l1_hinge == targets)
    #
    # l2_hinge = make_predictions(features, targets, 'hinge', 'l2')
    # # assert np.all(l2_hinge == targets)
    #
    # squared = make_predictions(features, targets, 'squared', None)
    # # assert np.all(squared == targets)
    #
    # l1_squared = make_predictions(features, targets, 'squared', 'l1')
    #
    # l2_squared = make_predictions(features, targets, 'squared', 'l2')

# test_squared_loss_forward()

def test_multiclass_gradient_descent_blobs():
    """
    Tests that the multiclass classifier also works on binary tasks
    """
    from your_code import MultiClassGradientDescent

    np.random.seed(0)

    features, _, targets, _ = load_data('blobs')

    learner = MultiClassGradientDescent(loss='squared', regularization=None,
                                        learning_rate=0.01, reg_param=0.05)
    learner.fit(features, targets, batch_size=None, max_iter=1000)
    predictions = learner.predict(features)

    print("predictions: ", predictions)
    print("targets: ", targets)


def FRQ_1():
    train_features, test_features, train_targets, test_targets = \
        load_data('mnist-binary', fraction=1.0)
    learner = GradientDescent( learning_rate=1e-4, loss='hinge')
    # print ("size: ", train_features.shape)
    # print (train_targets.shape)
    learner.fit(train_features, train_targets,  batch_size=1, max_iter=1000)
    # predictions = learner.predict(test_features)

def FRQ_2():
    train_features, test_features, train_targets, test_targets = \
    load_data('synthetic', fraction=1.0)
    bias_arr = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5,-5.5]

    indices = [0, 1, 4, 5]
    train_features = np.array(train_features)[indices[:]]
    train_targets = np.array(train_targets)[indices[:]]

    train_features = np.append(train_features, np.ones((train_features.shape[0],1)), axis=1)
    zero_one_loss = ZeroOneLoss()

    print("features: ", train_features)
    print("targets: ", train_targets)

    loss_arr = []
    for i_bias in range(len(bias_arr)):
        w = np.append( np.ones(train_features.shape[1] - 1), bias_arr[i_bias])
        loss = zero_one_loss.forward(train_features, w, train_targets)
        loss_arr.append(loss)

    plt.plot(bias_arr, loss_arr)
    plt.title("loss landscape")
    plt.xlabel("bias")
    plt.ylabel("loss")
    plt.show()

def FRQ_3():
    train_features, test_features, train_targets, test_targets = \
    load_data('mnist-multiclass', fraction=0.75)
    learner = MultiClassGradientDescent(loss='squared', regularization='l1')
    learner.fit(train_features, train_targets, batch_size=20, max_iter=700)
    predictions = learner.predict(test_features)

    print("predictionsL: ", predictions.shape)
    print("test_targets: ", test_targets.shape)
    c_m = confusion_matrix(test_targets, predictions.astype(int))
    visualized_c_m = np.append(np.arange(-1,5).reshape(1,6), np.append(np.arange(5).reshape(5,1), c_m, axis = 1),  axis=0 )
    print("confusion matrix: ", visualized_c_m)


def FRQ_4():
    train_features, test_features, train_targets, test_targets = \
    load_data('mnist-binary', fraction=1.0)
    print ("# of 0: ", np.sum([1 for target in train_targets if target==-1]))
    print ("# of 1: ", np.sum([1 for target in train_targets if target==1]))

    lambda_arr = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    l1_iter_arr = []
    l2_iter_arr = []
    l1_non_0_num_arr = []
    l2_non_0_num_arr = []

    batch_size = 50
    max_iter = 2000
    for _lambda in lambda_arr:
        l1_learner = GradientDescent( loss='squared', learning_rate=1e-5, regularization='l1', reg_param=_lambda)
        l2_learner = GradientDescent( loss='squared', learning_rate=1e-5, regularization='l2', reg_param=_lambda)

        l1_iter = l1_learner.fit(train_features, train_targets, batch_size=batch_size, max_iter=max_iter)
        l2_iter = l2_learner.fit(train_features, train_targets, batch_size=batch_size, max_iter=max_iter)

        l1_iter_arr.append(l1_iter)
        l2_iter_arr.append(l2_iter)

        l1_non_0_num = np.sum( [1 for i in l1_learner.model if abs(i)>=0.001 ] ) - 1
        l2_non_0_num = np.sum( [1 for i in l2_learner.model if abs(i)>=0.001 ] ) - 1

        l1_non_0_num_arr.append(l1_non_0_num)
        l2_non_0_num_arr.append(l2_non_0_num)

    print("l1 iterations: ", l1_iter_arr)
    print("l2 iterations: ", l1_iter_arr)

    plt.plot(np.log10(lambda_arr), l1_non_0_num_arr, label="L1")
    plt.plot(np.log10(lambda_arr), l2_non_0_num_arr, label="L2")
    plt.title("lambda values vs number of non zero weights with L1, L2 regularizer")
    plt.legend(loc="lower right")
    plt.xlabel("lambda ")
    plt.ylabel("Number of non zero weights")
    plt.show()

def FRQ_4_c():
    train_features, test_features, train_targets, test_targets = \
    load_data('mnist-binary', fraction=1.0)
    batch_size = 50
    max_iter = 2000
    l1_learner = GradientDescent( loss='squared', learning_rate=1e-5, regularization='l1', reg_param=1)
    l1_iter = l1_learner.fit(train_features, train_targets, batch_size=batch_size, max_iter=max_iter)

    data = np.copy(l1_learner.model)
    data = np.delete(data, -1)
    for i in range(len(data)):
        if abs(data[i])<0.001:
            data[i] = 0
        else:
            data[i] = 1
    data = data.reshape(28,28)

    plt.imshow(data,cmap='Greys')
    plt.title("heat map")
    plt.show()


FRQ_4_c()
