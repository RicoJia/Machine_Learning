"""
This module only contains the function that should be executed in parallel.
Since we want to run multiple cross validation experiments with our classifier
in parallel you will simply need to use the sklearn implementation for
cross validating an estimator.

You can find more about cross validation and model selection under this link:
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

Your implementation should return the mean score and standard deviation score for your cross validation
experiment and the associated parameters for the estimator
"""

from sklearn.model_selection import cross_val_score


def run_experiment(estimator, params, input_data, target_data):
    """
    This function is supposed to run a cross validation experiment with the given estimator
    and the given input and target dataset.

    This function should make use of the cross_val_score function specified by sklearn. Run CV
    with 20 different splits and calculate the mean score + std_score of all the scores that are generated
    by the cross_val_score function. This function should not be longer than 5 lines of src.

    :param estimator: The fully specified estimator that you want to run the experiment with
    :param input_data: The input dataset that you want to estimator to train on
    :param target_data: The target data for your input features
    what are params?

    :return: Should return the mean accuracy and standarddeviation of your accuracy of your estimator
        on the cross validation experiment and the associated parameters. The parameters are already
        given in the function call. You should return a tuple of (params, mean_score, std_score)
    """
    scores = cross_val_score(estimator, input_data, target_data, cv=20)
    mean = scores.mean()
    std = scores.std()
    num = input_data.shape[0]
    # return (params, mean, std, num)

    # test
    return (params, scores.tolist())


# def func(**kwargs) then you can pass in *(3,4,5) as a tuple, or **{"level": 3 , "blah": 3} as a dictionary
