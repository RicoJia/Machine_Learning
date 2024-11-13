import os
import random
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'parallellizer'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'worker'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'grid_search'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'mnist'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'circle'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'random_Search'))
from math import floor, log10

import matplotlib.pyplot as plt
import numpy as np
from experiment import run
from mnist import load_mnist
from scipy.stats import mannwhitneyu
from sklearn import svm


def round_sig(x, sig=2):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def frq_3():
    total_num = 0
    sample = None
    for digit in range(10):
        image_num = 0
        digit_ls = [digit]
        images, labels = load_mnist("testing", digit_ls)  # doctest: +SKIP
        image_num += images.shape[0]

        images, labels = load_mnist("training", digit_ls)  # doctest: +SKIP
        image_num += images.shape[0]

        print("digit: ", digit, "number of images: ", images.shape[0])
        total_num += image_num

    print("Total Size of Database: ", total_num)
    print("sample: ", sample)


def frq_9():
    train_images, train_labels = load_mnist(
        "training", selection=slice(0, 500)
    )  # doctest: +SKIP
    test_images, test_labels = load_mnist("testing", selection=slice(0, 500))
    random.seed(1)
    conditions = {"kernel": ["linear", "poly", "rbf"], "C": [0.1, 1, 10]}
    results = run(
        svm.SVC(gamma="auto"), "grid_search", conditions, train_images, train_labels
    )

    results = sorted(
        results, key=lambda result: list(result[0].values())[0], reverse=True
    )
    print("results: ", results)

    # # #build a table in which each row is kernel value, each column is slack value C. we know it's a 3x3 table
    for row in range(3):
        print_ls = []
        for cln in range(3):
            element = list(results.pop(0))
            for index in range(1, len(element)):
                element[index] = round_sig(element[index])
            print_ls.append(element)
        print(print_ls)


def frq_10():
    train_images, train_labels = load_mnist(
        "training", selection=slice(0, 500)
    )  # doctest: +SKIP
    test_images, test_labels = load_mnist("testing", selection=slice(0, 500))
    random.seed(1)
    conditions = {"kernel": ["linear", "poly", "rbf"], "C": [0.1, 1, 10]}
    # results = run(svm.SVC(gamma='auto'),
    #              "grid_search",
    #               conditions,
    #               train_images,
    #               train_labels)
    #
    # results = sorted(results, key=lambda result: list(result[0].values())[1], reverse=True)
    # print (results)

    results = [
        (
            {"kernel": "linear", "C": 10},
            [
                1.0,
                0.88,
                0.84,
                0.88,
                1.0,
                0.84,
                0.96,
                0.88,
                0.84,
                0.88,
                0.88,
                0.84,
                0.88,
                0.96,
                0.76,
                0.88,
                0.92,
                0.92,
                0.76,
                0.84,
            ],
        ),
        (
            {"kernel": "poly", "C": 10},
            [
                0.84,
                0.88,
                0.76,
                0.84,
                1.0,
                0.84,
                0.92,
                0.8,
                0.88,
                0.72,
                0.8,
                0.88,
                0.88,
                1.0,
                0.88,
                0.88,
                0.8,
                0.84,
                0.72,
                0.96,
            ],
        ),
        (
            {"kernel": "rbf", "C": 10},
            [
                1.0,
                0.92,
                0.84,
                0.88,
                1.0,
                0.88,
                0.96,
                0.96,
                0.88,
                0.88,
                0.92,
                0.8,
                0.96,
                0.96,
                0.92,
                0.88,
                0.88,
                0.96,
                0.84,
                0.88,
            ],
        ),
        (
            {"kernel": "linear", "C": 1},
            [
                1.0,
                0.88,
                0.84,
                0.88,
                1.0,
                0.84,
                0.96,
                0.88,
                0.84,
                0.88,
                0.88,
                0.84,
                0.88,
                0.96,
                0.76,
                0.88,
                0.92,
                0.92,
                0.76,
                0.84,
            ],
        ),
        (
            {"kernel": "poly", "C": 1},
            [
                0.8,
                0.72,
                0.76,
                0.84,
                0.96,
                0.84,
                0.92,
                0.76,
                0.8,
                0.72,
                0.8,
                0.84,
                0.88,
                1.0,
                0.8,
                0.88,
                0.8,
                0.88,
                0.6,
                0.92,
            ],
        ),
        (
            {"kernel": "rbf", "C": 1},
            [
                0.96,
                0.88,
                0.88,
                0.92,
                1.0,
                0.88,
                0.96,
                0.96,
                0.84,
                0.84,
                0.84,
                0.8,
                0.88,
                0.96,
                0.88,
                0.92,
                0.84,
                0.96,
                0.8,
                0.88,
            ],
        ),
        (
            {"kernel": "linear", "C": 0.1},
            [
                1.0,
                0.88,
                0.84,
                0.88,
                1.0,
                0.84,
                0.96,
                0.88,
                0.84,
                0.88,
                0.88,
                0.84,
                0.88,
                0.96,
                0.76,
                0.88,
                0.92,
                0.88,
                0.76,
                0.84,
            ],
        ),
        (
            {"kernel": "poly", "C": 0.1},
            [
                0.6,
                0.56,
                0.56,
                0.6,
                0.72,
                0.64,
                0.64,
                0.6,
                0.52,
                0.44,
                0.68,
                0.68,
                0.76,
                0.68,
                0.56,
                0.6,
                0.56,
                0.68,
                0.4,
                0.48,
            ],
        ),
        (
            {"kernel": "rbf", "C": 0.1},
            [
                0.56,
                0.56,
                0.64,
                0.64,
                0.64,
                0.56,
                0.68,
                0.56,
                0.52,
                0.48,
                0.48,
                0.56,
                0.6,
                0.68,
                0.6,
                0.6,
                0.48,
                0.56,
                0.44,
                0.44,
            ],
        ),
    ]

    # data = [c10, c1, c01]
    #
    f1 = plt.figure(1)
    list1 = [i[1] for i in results[0:3]]  # 3x3
    c10 = [item for sublist in list1 for item in sublist]

    list2 = [i[1] for i in results[3:6]]  # 3x3
    c1 = [item for sublist in list2 for item in sublist]

    list3 = [i[1] for i in results[6:9]]  # 3x3
    c01 = [item for sublist in list3 for item in sublist]

    w10_1, p10_1 = mannwhitneyu(c10, c1)
    w10_01, p10_01 = mannwhitneyu(c10, c01)
    w1_01, p1_01 = mannwhitneyu(c1, c01)
    print("p10_1: ", p10_1)
    print("p10_01: ", p10_01)
    print("p1_01: ", p1_01)
    # bpl3 = plt.boxplot([c10, c1, c01], positions=np.arange(0,3), widths=0.6)
    # ticks = ['10', '1', '0.1']
    # plt.xticks(np.arange(0,3), ticks)
    # plt.xlabel("Kernal Value")
    # plt.ylabel("Cross Validation Score")
    # plt.title("n = 500 accuracy plot, Varying C")
    #
    #
    #
    # f4 = plt.figure(4)
    # list1 = [i[1] for i in results[0:3]]      #3x3
    # c10 = [item for sublist in list1 for item in sublist]
    # plt.hist(c10, normed=True, bins=100)
    # plt.ylabel('Cross Validation Score Distribution')
    # plt.title("n = 500 accuracy plot, C = 10")
    #
    # f5 = plt.figure(5)
    # list2 = [i[1] for i in results[3:6]]      #3x3
    # c1 = [item for sublist in list2 for item in sublist]
    # plt.hist(c1, normed=True, bins=100)
    # plt.ylabel('Cross Validation Score Distribution')
    # plt.title("n = 500 accuracy plot, C = 1")
    #
    # f6 = plt.figure(6)
    # list3 = [i[1] for i in results[6:9]]      #3x3
    # c01 = [item for sublist in list3 for item in sublist]
    # plt.hist(c01, normed=True, bins=100)
    # plt.ylabel('Cross Validation Score Distribution')
    # plt.title("n = 500 accuracy plot, C = 01")
    # plt.show()


def frq_13():

    # train_images, train_labels = load_mnist('training', selection=slice(0, 500)) # doctest: +SKIP
    # test_images, test_labels = load_mnist('testing', selection=slice(0, 500))
    # random.seed(1)
    # conditions = {"kernel": ["linear","poly", "rbf"], "C": [0.1, 1, 10 ]}
    # results = run(svm.SVC(gamma='auto'),
    #              "grid_search",
    #               conditions,
    #               train_images,
    #               train_labels)
    #
    # results = sorted(results, key=lambda result: list(result[0].values())[0], reverse=True)
    # print (results)

    results = [
        (
            {"kernel": "rbf", "C": 0.1},
            [
                0.56,
                0.56,
                0.64,
                0.64,
                0.64,
                0.56,
                0.68,
                0.56,
                0.52,
                0.48,
                0.48,
                0.56,
                0.6,
                0.68,
                0.6,
                0.6,
                0.48,
                0.56,
                0.44,
                0.44,
            ],
        ),
        (
            {"kernel": "rbf", "C": 1},
            [
                0.96,
                0.88,
                0.88,
                0.92,
                1.0,
                0.88,
                0.96,
                0.96,
                0.84,
                0.84,
                0.84,
                0.8,
                0.88,
                0.96,
                0.88,
                0.92,
                0.84,
                0.96,
                0.8,
                0.88,
            ],
        ),
        (
            {"kernel": "rbf", "C": 10},
            [
                1.0,
                0.92,
                0.84,
                0.88,
                1.0,
                0.88,
                0.96,
                0.96,
                0.88,
                0.88,
                0.92,
                0.8,
                0.96,
                0.96,
                0.92,
                0.88,
                0.88,
                0.96,
                0.84,
                0.88,
            ],
        ),
        (
            {"kernel": "poly", "C": 0.1},
            [
                0.6,
                0.56,
                0.56,
                0.6,
                0.72,
                0.64,
                0.64,
                0.6,
                0.52,
                0.44,
                0.68,
                0.68,
                0.76,
                0.68,
                0.56,
                0.6,
                0.56,
                0.68,
                0.4,
                0.48,
            ],
        ),
        (
            {"kernel": "poly", "C": 1},
            [
                0.8,
                0.72,
                0.76,
                0.84,
                0.96,
                0.84,
                0.92,
                0.76,
                0.8,
                0.72,
                0.8,
                0.84,
                0.88,
                1.0,
                0.8,
                0.88,
                0.8,
                0.88,
                0.6,
                0.92,
            ],
        ),
        (
            {"kernel": "poly", "C": 10},
            [
                0.84,
                0.88,
                0.76,
                0.84,
                1.0,
                0.84,
                0.92,
                0.8,
                0.88,
                0.72,
                0.8,
                0.88,
                0.88,
                1.0,
                0.88,
                0.88,
                0.8,
                0.84,
                0.72,
                0.96,
            ],
        ),
        (
            {"kernel": "linear", "C": 0.1},
            [
                1.0,
                0.88,
                0.84,
                0.88,
                1.0,
                0.84,
                0.96,
                0.88,
                0.84,
                0.88,
                0.88,
                0.84,
                0.88,
                0.96,
                0.76,
                0.88,
                0.92,
                0.88,
                0.76,
                0.84,
            ],
        ),
        (
            {"kernel": "linear", "C": 1},
            [
                1.0,
                0.88,
                0.84,
                0.88,
                1.0,
                0.84,
                0.96,
                0.88,
                0.84,
                0.88,
                0.88,
                0.84,
                0.88,
                0.96,
                0.76,
                0.88,
                0.92,
                0.92,
                0.76,
                0.84,
            ],
        ),
        (
            {"kernel": "linear", "C": 10},
            [
                1.0,
                0.88,
                0.84,
                0.88,
                1.0,
                0.84,
                0.96,
                0.88,
                0.84,
                0.88,
                0.88,
                0.84,
                0.88,
                0.96,
                0.76,
                0.88,
                0.92,
                0.92,
                0.76,
                0.84,
            ],
        ),
    ]

    # data = [c10, c1, c01]
    #
    # f1= plt.figure(1)
    list1 = [i[1] for i in results[0:3]]  # 3x3
    c10 = [item for sublist in list1 for item in sublist]

    list2 = [i[1] for i in results[3:6]]  # 3x3
    c1 = [item for sublist in list2 for item in sublist]

    list3 = [i[1] for i in results[6:9]]  # 3x3
    c01 = [item for sublist in list3 for item in sublist]
    #
    w10_1, p10_1 = mannwhitneyu(c10, c1)
    w10_01, p10_01 = mannwhitneyu(c10, c01)
    w1_01, p1_01 = mannwhitneyu(c1, c01)
    print("p10_1: ", p10_1)
    print("p10_01: ", p10_01)
    print("p1_01: ", p1_01)

    # bpl3 = plt.boxplot([c10, c1, c01], positions=np.arange(0,3), widths=0.6)
    # ticks = [list((results[0][0]).values())[0], list((results[3][0]).values())[0],list((results[6][0]).values())[0]]
    # plt.xticks(np.arange(0,3), ticks)
    # plt.xlabel("Kernel Value")
    # plt.ylabel("Cross Validation Score")
    # plt.title("n = 500 accuracy plot, Varying Kernel")
    #
    #
    #
    # f4 = plt.figure(4)
    # list1 = [i[1] for i in results[0:3]]      #3x3
    # c10 = [item for sublist in list1 for item in sublist]
    # plt.hist(c10, normed=True, bins=100)
    # plt.ylabel('Cross Validation Score Distribution')
    # plt.title("n = 500 accuracy plot - kernel = rbf")
    # #
    # f5 = plt.figure(5)
    # list2 = [i[1] for i in results[3:6]]      #3x3
    # c1 = [item for sublist in list2 for item in sublist]
    # plt.hist(c1, normed=True, bins=100)
    # plt.ylabel('Cross Validation Score Distribution')
    # plt.title("n = 500 accuracy plot - kernel = poly")
    # #
    # f6 = plt.figure(6)
    # list3 = [i[1] for i in results[6:9]]      #3x3
    # c01 = [item for sublist in list3 for item in sublist]
    # plt.hist(c01, normed=True, bins=100)
    # plt.ylabel('Cross Validation Score Distribution')
    # plt.title("n = 500 accuracy plot - kernel = linear")
    # plt.show()


frq_10()
