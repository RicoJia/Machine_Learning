
# Important background

## Scikit learn

You will use [scikit-learn](http://scikit-learn.org/stable/index.html), a machine learning library for Python, to answer questions in this homework. 
You should be running the latest stable version of scikit-learn (0.21.3, as of this writing).
If you want an example of how to train and call a classifier from scikit-learn, have a look at the [main page for the support vector machine](http://scikit-learn.org/stable/modules/svm.html#multi-class-classification).
Most classifiers have similarly good documentation and are called in similar ways.
For easy-to-use model selection, cross validation, etc, check out [the documentation on model selection](http://scikit-learn.org/stable/model_selection.html#model-selection)

## SciPy and statistical tests
Here are some helpful statistical tests to determine whether two samples are drawn from the same underlying distribution.

* If you have paired samples and normally distributed data, use this: [paired samples t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)

* If you have independent samples and normally distributed data, use this: [independent samples t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

* If you have paired samples and data that doesn't follow a normal distribution use this: [Wilcoxon signed-rank test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html?highlight=wilcoxon#scipy.stats.wilcoxon)

* If you have independent samples and data that doesn't follow a normal distribution use this:[Mann–Whitney U test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html?highlight=mannwhitney#scipy.stats.mannwhitneyu)


## The MNIST dataset
The MNIST dataset of handwritten digits is used for this assignment. You can read more about it [here](http://yann.lecun.com/exdb/mnist/). We've provided a data loader for you in `mnist.py`, but you must download the dataset for yourself. We've provided a data loader for you in `mnist.py`, but you must download and extract the dataset for yourself. Make sure you download all four files (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, and `t10k-labels-idx1-ubyte.gz`). Instructions for extracting `.gz` files can be found for Windows and Mac [here](https://www.wikihow.com/Extract-a-Gz-File), and Unix instructions are [here](https://askubuntu.com/questions/25347/what-command-do-i-need-to-unzip-extract-a-tar-gz-file). Do not push these files to your github repository. You will need to use the data loader for some questions below. Please add these files to the data folder!

## The CIRCLE dataset
The CIRCLE dataset is a handcrafted dataset which shows a circle within another circle. That means, that this dataset is not separable by a linear hyperplane (At least without applying a special kernel to it). You can find a function `load_circle()` in the file `circle.py` which will load this dataset. This dataset is only there for the test cases and the coding part of the assignment. You will only need the MNIST dataset to answer the free response questions.

# Coding (5 points)

The main task for this assignment will be building two hyperparameter tuners. Your code for the GridSearchCV tuner should fit into `grid_search.py` while the RandomSearchCV code goes into the `random_search.py` script. Both files already contain an outline for your code. You can find more details about hyperparameter tuning and the mentioned algorithms [here](https://en.wikipedia.org/wiki/Hyperparameter_optimization).

To boost the performance of your code we also included a parallelizer that can run mutliple training jobs or worker tasks on different threads at the same time. You can find the code for that in `parallelizer.py`. Read through the comments of this file so that you understand how the parallelization works. You can initialize a new parallelizer object with a worker function. This function and its signature is already given in `worker.py`.

The test cases rely on the code that is already given in `experiment.py`. The only thing that you will need to change is the amount of MNIST data which we want to use during the training time. If we would use the whole MNIST dataset it takes several hours to train a single Support Vector Machine. Since we want to run mutliple experiments within our hyperparameter tuner, you need to change the value of the NUMBER_OF_MNIST_SAMPLES variable so that a single SVM training experiment including cross validation does not take longer than 2 minutes. In our case this number was set to 2000 but feel free to change the number if you want to.
 
You should make a conda environment for this homework just like you did for previous homeworks. We have included a `requirements.txt`.

# Free-response questions (5 points)

#### Understanding SVMs (1 point)
1\. (0.5 points) Explain why a support vector machine using a kernel, once trained, does not directly use the decision boundary to classify points. 

2\. (0.5 points) If the support vector machine does not directly use the decision boundary to classify points, how does it, in fact, classify points. *Hint, what are the support vectors?*

#### the MNIST data (1 point)
3\. (0.5 points) How many images are there in the MNIST data? How many images are there of each digit? How many different people's handwriting? Are the digit images all the same size and orientation? What is the color palette of MNIST (grayscale, black & white, RGB)?

4\. (0.5 points) Select one of the digits from the MNIST data. Look through the variants of this digit that different people produced. Show us 3 examples of that digit you think might be challenging for a classifier to correctly classify. Explain why you think they might be challenging.

#### Selecting training and testing data  (.5 points)

5\. (0.5 points) Now you have to decide how to make a draw from the data for training and testing a model. Think about the goals of training and testing sets - we pick good training sets so our classifier generalizes to unseen data and we pick good testing sets to see whether our classifier generalizes. Explain how you should select training and testing sets. (Entirely randomly? Train on digits 0-4, test on 5-9? Train on one group of hand-writers, test on another?). Justify your method for selecting the training and testing sets in terms of these goals. 

#### Finding the best hyperparameters (2.5 points)
To answer the following questions you should use your GridSearch hyperparameter tuner. We want to find the best kernel and slack cost, **C**, for handwritten digit recognition on MNIST using a support vector machine. To do this, we're going to try different kernels from the set {Linear, Polynomial, Radial Basis Function}. Use the default value of 3 for the degree of the polynomial. We will combine each kernel with a variety of **C** values drawn from the set { 0.1, 1, 10 }. This results in 9 variants of the SVM. For each variant we will be running 20 fold cross validation which was specified in the `worker.py`. You can simply call the run function within `experiment.py` with the right parameters to get all the results that you nedd.

9\. (0.5 point) Create a table with 3 rows (1 kernel per row) and 3 columns (the 3 slack settings). Rows and columns should be clearly labeled. For each condition (combination of slack and kernel), show the following 3 values: the mean accuracy **a** of the trials, the standard deviation of the accuracy **std** and the number of trials/experiments **n**, written in the format: **mean(a),std(a),n**. 

10\. (0.5 points) Make a boxplot graph that plots accuracy (vertical) as a function of the slack **C** . There should be 3 boxplots in the graph, one per value of **C**. Use results across all kernels. Indicate **n** on your plot, where **n** is the number of trials per boxplot. Don't forget to label your dimensions. 

11\. (0.25 points) What statistical test should you use to do comparisons between the values of **C** plotted in the previous question? Explain the reason for your choice. Consider how you selected testing and training sets and the skew of the data in the boxplots in your answer. _Note: Your boxplots will show you whether a distribution is skewed (and thus, not normal), but will not show you what the shape of each distribution. There are distributions that are not skewed, but are still not bell curves (normal distributions). It would be a good idea to look at the histograms of your distributions to decide which statistical test you should use._

12\. (0.25 points) Give the p value reported by your test. Say what that p value means with respect to the 5% rule. 

13\. (0.5 points) Make a boxplot graph that plots accuracy (vertical) as a function of __kernel__ choice. There should be 3 boxplots in the graph, one per kernel. Use results across all values for C. Don't forget to indicate **n** on your plot, where **n** is the number trials per boxplot. Don't forget to label your dimensions. 

14\. (0.25 points) What statistical test should you use to determine whether the difference between the best and second best kernel is statistically significant? Explain the reason for your choice. Consider how you selected testing and training sets and the skew of the data in the boxplots in your answer. 

15\. (0.25 points) What is the result of your statistical test? Is the difference between the best and second best value of kernel statistically significant?