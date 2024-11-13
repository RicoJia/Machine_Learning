import numpy as np


def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [                                   predicted|actual
        [true_negatives, false_positives],[--, +-]
        [false_negatives, true_positives] [-+, ++]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO
    CLASSES (binary).

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N (test_targets)  Nx1 vec
        predictions (np.array): predicted labels of length N    (predicted result from code)    Nx1 vec

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    true_total = np.equal(actual, predictions).sum()
    true_positives = len(
        [
            i
            for i in range(len(actual))
            if (actual[i] == predictions[i]) and (actual[i] > 0)
        ]
    )
    true_negatives = true_total - true_positives

    false_total = len(actual) - true_total
    false_positives = len(
        [
            i
            for i in range(len(actual))
            if (actual[i] != predictions[i]) and (predictions[i] > 0)
        ]
    )
    false_negatives = false_total - false_positives

    return np.array(
        [[true_negatives, false_positives], [false_negatives, true_positives]]
    )


def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    confusion_mat = confusion_matrix(actual, predictions)
    true_positives = confusion_mat[1][1]
    false_positives = confusion_mat[0][1]
    false_negatives = confusion_mat[1][0]
    true_negatives = confusion_mat[0][0]
    acc = (true_positives + true_negatives) / (
        true_positives + true_negatives + false_negatives + false_positives
    )
    return acc


def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confusion_mat = confusion_matrix(actual, predictions)
    true_positives = confusion_mat[1][1]
    false_positives = confusion_mat[0][1]
    false_negatives = confusion_mat[1][0]
    true_negatives = confusion_mat[0][0]

    if false_positives + true_positives == 0:
        precision = 0.0  # this happens
    else:
        precision = true_positives / (false_positives + true_positives)
    recall = true_positives / (false_negatives + true_positives)

    return precision, recall


def f1_measure(actual, predictions):
    """
     Given predictions (an N-length numpy vector) and actual labels (an N-length
     numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

     Hint: implement and use the precision_and_recall function!

     Args:
         actual (np.array): predicted labels of length N
         predictions (np.array): predicted labels of length N

     Output:
         f1_measure (float): F1 measure of dataset (harmonic mean of precision and
         recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)

    if precision + recall != 0:
        F1 = (2.0 * precision * recall) / (precision + recall)
    else:
        F1 = 0
    return F1
