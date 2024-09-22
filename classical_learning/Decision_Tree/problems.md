# Coding (5 points)
Your task is to implement two machine learning algorithms:

1. Decision tree (in `code/decision_tree.py`)
2. Prior probability (in `code/prior_probability.py`)

You will also write code that reads in data into numpy arrays and code that manipulates
data for training and testing in `code/data.py`.

You will implement evaluation measures in `code/metrics.py`:

1. Confusion matrix (`code/metrics.py -> confusion_matrix`)
2. Precision and recall (`code/metrics.py -> precision_and_recall`)
3. F1-Measure (`code/metrics.py -> f1_measure`)

The entire workflow will be encapsulated in `code/experiment.py -> run`. The run function 
will allow you to run each approach on different datasets easily. You will have to 
implement this `run` function.

Your goal is to pass the test suite (contained in `tests/`). Once the tests are passed, you 
may move on to the next part - reporting your results.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here. Suggested order for passing test_cases:

1. test_load_data
2. test_train_test_split
3. test_confusion_matrix
4. test_accuracy
5. test_precision_and_recall
6. test_f1_measure
7. test_experiment_run_prior_probability
8. test_experiment_run_decision_tree
9. test_experiment_run_and_compare

# Free-response questions (5 points)

To answer some of these questions you will have to read the assigned reading (Chapter 3 of Machine learning) for decision trees. 

To answer some of these questions, you will have to write extra code (that is not covered by the test cases). The extra code should import your implementation and run experiments on the various datasets (e.g. choosing `ivy-league.csv` for a dataset and doing `experiment.run` with a 80/20 train/test split, averaged over a few trials).

1. Assume you have a deterministic function that takes a fixed, finite number of Boolean inputs and returns a Boolean output. Can a decision tree be built to represent any such function? Give a simple proof for your answer. 

2. Let <img src="/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/>  be a voter in the set of voters <img src="/tex/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode&sanitize=true" align=middle width=13.242037049999992pt height=22.465723500000017pt/>. Let each <img src="/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/> have a value, assigned to either -1 or 1. Let's define Majority-rule as the sign of the sum all voters. If the sum is negative, it returns -, else it returns +. Assume an odd number of voters (so there are no ties). Can the majority-rule algorithm be represented by a decision tree that considers a single voter at each decision node? Why or why not?

3. In the coding section of this assignment, you trained a decision tree with the ID3 algorithm on several datasets (`candy-data.csv`, `majority-rule.csv`, `ivy-league.csv`, and `xor.csv`). For each dataset, report the accuracy on the testing data, the number of nodes in the tree and the maximum depth (number of levels) of the tree. 

4. What is the bias of the ID3 algorithm in the way it searches the hypothesis space of possible decision trees?

5. What is the the thing that `majority-rule.csv` and `xor.csv` have in common? How does this thing interact with the bias in the way ID3 builds a decision tree? 

6. Explain what overfitting is and describe how one can tell it is happening.

7. Explain how Reduced Error Pruning is done, the reason for doing it, and what the tradeoffs are in applying Reduced Error Pruning.

8. One can modify the simple ID3 algorithm to handle attributes that are real-valued (e.g. height, weight, age). To do this, one must pick a split point for each attribute (e.g. height > 3) and then determine information gain, given the split point. How would you pick a split point automatically? Why would you do it that way?  

9. Assume each person in a population is has two real-valued measured attributes: height, weight. (Also assume we live in a universe in which height and weight have the same units and are directly comparable.) In a two-dimensional plot, illustrate the decision boundary line for the concept <img src="/tex/a293d914d054946fddd8fb36eadc4799.svg?invert_in_darkmode&sanitize=true" align=middle width=117.90945869999997pt height=22.831056599999986pt/>. Those with <img src="/tex/a293d914d054946fddd8fb36eadc4799.svg?invert_in_darkmode&sanitize=true" align=middle width=117.90945869999997pt height=22.831056599999986pt/>  get a 0, those with <img src="/tex/3ad4ece50c40480076bd450203ac449d.svg?invert_in_darkmode&sanitize=true" align=middle width=117.90945869999997pt height=22.831056599999986pt/> get a 1.  Now, assume you have a decision tree that uses a single real-valued attribute (plus an ideally-chosen split point value) at each decision node. Can you represent the concept <img src="/tex/a293d914d054946fddd8fb36eadc4799.svg?invert_in_darkmode&sanitize=true" align=middle width=117.90945869999997pt height=22.831056599999986pt/> with such a tree?  In other words, you want to always return 1 if someone's weight is greater than their height and 0 otherwise.  Can you do it? If so, specify the decision tree. If not, say why not.

10. Ensemble methods are learning methods that combine the output of multiple learners. The hope is that an ensemble will do better than any one learner, because it reduces the overfitting a single learner may be suceptible to. One of the most popular ensemble methods is the random forest. The high level idea is to build multiple decision trees from the training data. The way one builds different decision trees from the same data is to train decision trees on different random subsets of the attributes. If, for example, there are 20 measurable attributes, you randomly pick 10 of the attributes and build a tree on those 10, then you randomly pick another 10 attributes and build a tree using those 10 attributes. If you were building an ensemble of <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> trees this way, how would you combine their decisions in the end? Explain why you would choose this method. Feel free to provide a citation for your choice (if you cite something, please ALSO provide a hyperlink), but also explain the reason this is your choice.





