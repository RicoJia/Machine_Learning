import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


class PolynomialRegression:
    def __init__(self, degree):
        """
        Implement polynomial regression from scratch.

        This class takes as input "degree", which is the degree of the polynomial
        used to fit the data. For example, degree = 2 would fit a polynomial of the
        form:

            ax^2 + bx + c

        Your code will be tested by comparing it with implementations inside sklearn.
        DO NOT USE THESE IMPLEMENTATIONS DIRECTLY IN YOUR CODE. You may find the
        following documentation useful:

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        Here are helpful slides:

        http://interactiveaudiolab.github.io/teaching/eecs349stuff/eecs349_linear_regression.pdf

        The internal representation of this class is up to you. Read each function
        documentation carefully to make sure the input and output matches so you can
        pass the test cases. However, do not use the functions numpy.polyfit or numpy.polval.
        You should implement the closed form solution of least squares as detailed in slide 10
        of the lecture slides linked above.

        Usage:
            import numpy as np

            x = np.random.random(100)
            y = np.random.random(100)
            learner = PolynomialRegression(degree = 1)
            learner.fit(x, y) # this should be pretty much a flat line
            predicted = learner.predict(x)

            new_data = np.random.random(100) + 10
            predicted = learner.predict(new_data)

        Args:
            degree (int): Degree of polynomial used to fit the data.
        """
        self.degree = degree
        self.w = None
        self.polyfit = None

    def fit(self, features, targets):
        """
        Fit the given data using a polynomial. The degree is given by self.degree,
        which is set in the __init__ function of this class. The goal of this
        function is fit features, a 1D numpy array, to targets, another 1D
        numpy array.


        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.(y)
        Returns:
            None (saves model and training data internally)
        """

        X = np.empty((0, self.degree + 1))
        for x in features:
            x_row = np.array([1.0])
            x0 = x
            for n in range(self.degree):
                x_row = np.append(x_row, x0)
                x0 = x0 * x
            # #test: <1, x^n...x>
            # x_row_new = np.append(1, (x_row[::-1])[:-1])
            X = np.vstack((X, x_row))

        # # X_unique_cln = np.unique(X, axis=1)
        # np_polyfit = np.polyfit(features, targets, self.degree)
        #
        # # print("----X: ", X)
        # # print("Features: ", features)
        # # self.polyfit = np_polyfit[::-1]
        # print ("poly fit: ",np_polyfit)
        # w_vec = (np.linalg.inv((X_unique_cln.T).dot(X_unique_cln)).dot(X_unique_cln.T)).dot(targets)
        # w_vec_new = np.append( (w_vec[:-1][::-1]), w_vec[-1] )
        # # self.w = w_vec_[::-1]
        # # print("my result: ", w_vec)
        # self.w = w_vec_new[::-1]
        # print("my result: ", w_vec_new)

        # test
        X_unique_cln = X
        XTX = (X_unique_cln.T).dot(X_unique_cln)

        # if np.linalg.matrix_rank(XTX)!= XTX.shape[0]:
        #     w_vec = (np.linalg.pinv(XTX).dot(X_unique_cln.T)).dot(targets)
        # else:
        #     w_vec = (np.linalg.inv(XTX).dot(X_unique_cln.T)).dot(targets)
        w_vec = (np.linalg.inv(XTX).dot(X_unique_cln.T)).dot(targets)
        self.w = w_vec  # QUESTION: WHAT THE FUCK!!! np.unique doesn't work!, pseudo inverse does, even doing nothing can pass the test!!

    def predict(self, features):
        """
        Given features, a 1D numpy array, use the trained model to predict target
        estimates. Call this after calling fit.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        predictions = np.array([])
        for x in features:
            x_poly = np.array([1])  # <1,x...x^deg>
            for n in range(self.degree):
                x_poly = np.append(x_poly, x_poly[-1] * x)
            y_val = (self.w).dot(x_poly)
            predictions = np.append(predictions, y_val)

        return predictions

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the polynomial fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION. Instead, use plt.savefig().

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (plots to the active figure)
        """
        plt.scatter(features, targets, c="r", marker="o")
        plt.plot(features, self.predict(features))
        plt.savefig("polynomial_regrression.png")


# from generate_regression_data import generate_regression_data
# from metrics import mean_squared_error
##free response Q1
# x, y = generate_regression_data(4, 100, amount_of_noise=0.2)
#
# train_features = x[:50]
# train_targets = y[:50]
# test_features = x[50:]
# test_targets = y[50:]
#
# errors = np.zeros(10)
# errors_train = np.zeros(10)
# for deg in range(10):
#    learner = PolynomialRegression(degree = deg)
#    learner.fit(train_features, train_targets) # this should be pretty much a flat line
#    predictions = learner.predict(test_features)    # test features
#    errors[deg] = mean_squared_error(predictions, test_targets)
#
#    trained_predictions = learner.predict(train_features)
#    errors_train[deg] = mean_squared_error(trained_predictions, train_targets)
#
#
## # question 1
# plt.plot(np.linspace(0,9,10),errors, label='test error')
# plt.plot(np.linspace(0,9,10),errors_train, label='train error')
# plt.title('Errors')
# plt.xlabel('Features')
# plt.ylabel('Error')
# plt.legend()
# plt.show()
#
# train_lowest_error_i = list(errors_train).index(min(errors_train))
# test_lowest_error_i = list(errors).index(min(errors))
## print("train lowest error: ", train_lowest_error_i, "test lowest error: ", test_lowest_error_i )
# test_best_learner = PolynomialRegression(degree=test_lowest_error_i)
# test_best_learner.fit(train_features, train_targets)
# train_best_learner = PolynomialRegression(degree=train_lowest_error_i)
# train_best_learner.fit(train_features, train_targets)
# x= np.linspace(min(train_features), max(train_features))
# y_test = test_best_learner.predict(x)
# y_train = train_best_learner.predict(x)
#
##test
## y_retrained = train_best_learner.predict(train_features)
## print ("errors: ", errors_train," train lowest error: ", train_lowest_error_i, "new errors after relearning: ",mean_squared_error(y_retrained, train_targets))
## print ("ytrain: ", y_train)
#
## plt.plot(np.linspace(0,9,10), errors, label= 'Train Error')
## plt.plot(np.linspace(0,9,10), errors_train, label='Test Error')
# plt.title('Question 1 A errors')
# plt.xlabel('Feature Value')
# plt.ylabel('Y value (Response and Predicted Values)')
#
# high_list_x1 = []
# high_target_y = []
# low_list_x1 = []
# low_target_y = []
# for index, target in np.ndenumerate(train_targets):
#    if target > 0:
#        high_list_x1.append(train_features[index])
#        high_target_y.append(train_targets[index])
#    else:
#        low_list_x1.append(train_features[index])
#        low_target_y.append(train_targets[index])
#
# plt.scatter(high_list_x1,high_target_y, c='b',marker='o')
# plt.scatter(low_list_x1,low_target_y, c='r',marker='+')
#
# print ("high_list: ", high_list_x1)
## plt.scatter(train_features, train_targets, c='r')
# plt.plot(x, y_test, c = 'b', label='Lowest test error polynomial')
# plt.plot(x, y_train, label='Lowest train error polynomial')
# plt.title('Scatter Plot and Fitting Curves')
## plt.yscale('log')
# plt.legend()
## plt.yticks(np.linspace(min(y_train), max(np.append(y_test, y_train)), 10))
# plt.show()
