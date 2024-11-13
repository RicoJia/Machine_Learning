import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT
    in general).

    Args:
        features (np.ndarray): input features[[x1,x2], [x1,x2]]
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    # since the data we're transforming is circle, we 1. find center 2. map it to theta and r space
    means = np.mean(features, axis=0)
    x_center = means[0]
    y_center = means[1]

    trans_features = np.empty((0, 2))
    for row in features:
        trans_row = np.zeros(2)
        trans_row[0] = row[0]
        trans_row[1] = (row[0] - x_center) ** 2 + (row[1] - y_center) ** 2

        trans_features = np.vstack((trans_features, trans_row))

    return trans_features


class Perceptron:
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the
        line are one class and points on the other side are the other class.

        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm.

        Args:
            max_iterations (int): the perceptron learning algorithm stops after
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.w = None

    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for
        the perceptron learning algorithm:

        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end

        Args:
            features (np.ndarray): 2D array containing inputs. [x11, x12], [x21, x22], ...
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        # w_vec = np.append(1, features[0])*targets[0]
        # iteration = 0
        # num_satisfied_features = 1
        # m = features.shape[0]
        # while iteration<self.max_iterations and num_satisfied_features<m:
        #     k = (iteration+1)%m
        #     x_preppended = np.append(1,features[k])   #<1,x1,x2>
        #     h_k = 1 if w_vec.dot(x_preppended)>0 else -1
        #     if h_k != targets[k]:  #violating the g(x)
        #         w_vec+=x_preppended*targets[k]
        #         num_satisfied_features = 0
        #     iteration+=1

        w_vec = np.zeros(features.shape[1] + 1)
        iteration = 0

        m = features.shape[0]
        while iteration < self.max_iterations:
            sum_error = 0.0
            for k in range(m):
                x_preppended = np.append(1, features[k])  # <1,x1,x2>
                h_k = 1 if w_vec.dot(x_preppended) > 0 else -1
                error = targets[k] - h_k  # 0,2,-2
                sum_error += (h_k - targets[k]) ** 2 / 4.0
                if h_k != targets[k]:  # violating the g(x)
                    w_vec += x_preppended * error / 2
            iteration += 1

        self.w = w_vec

        # Notes: 1. I got mine working not stricly following the algorithm on the book. I did w = w+x*error, not w+w*y.  2. x_preppended should be <1,x>, that's the algorithm.

    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        features_appended = np.append(
            (np.ones(features.shape[0]))[:, np.newaxis], features, axis=1
        )
        predicted = np.sign(features_appended.dot(self.w))
        return predicted

    def visualize(self, features, targets, name):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        high_list_x1 = []
        high_list_x2 = []
        low_list_x1 = []
        low_list_x2 = []
        for index, target in np.ndenumerate(targets):
            if target == 1:
                high_list_x1.append(features[index][0])
                high_list_x2.append(features[index][1])
            else:
                low_list_x1.append(features[index][0])
                low_list_x2.append(features[index][1])

        # high_list_x1 = []
        # high_list_x2 = []
        # low_list_x1 = []
        # low_list_x2 = []
        # for index, target in np.ndenumerate(self.predict(features)):
        #     if target == 1:
        #         high_list_x1.append(features[index][0])
        #         high_list_x2.append(features[index][1])
        #     else:
        #         low_list_x1.append(features[index][0])
        #         low_list_x2.append(features[index][1])

        # plt.scatter(high_list_x1,high_list_x2, c='b',marker='o')
        # plt.scatter(low_list_x1,low_list_x2, c='r',marker='+')
        # k = -self.w[1]/self.w[2]
        # b = -self.w[0]/self.w[2]
        # x1 = features[1]
        # x2 = k*features[2]+b
        # plt.xlabel('x1')
        # plt.ylabel('x2')
        # plt.plot(x1,x2)
        # plt.show()
        # plt.savefig("perceptron.png")

        plt.scatter(high_list_x1, high_list_x2, c="b", marker="o")
        plt.scatter(low_list_x1, low_list_x2, c="r", marker="+")
        k = -self.w[1] / self.w[2]
        b = -self.w[0] / self.w[2]
        x1 = np.sort(features[:, 0])
        x2 = k * x1 + b
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.plot(x1, x2)
        plt.title("Perceptron Classification - " + name)
        plt.show()


# from load_json_data import load_json_data
# import os
# data_files = [
#    os.path.join('../data', x)
#    for x in os.listdir('../data/')
#    if x[-4:] == 'json']
# for name in data_files:
#    features, targets = load_json_data(name)
#    p = Perceptron(max_iterations=100)
#
#
#    p.fit(features, targets)
#    # targets_hat = p.predict(features)
#    # features = transform_data(features)
#    p.visualize(features, targets, name)
