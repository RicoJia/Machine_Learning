import csv
import os

import numpy as np


def read_entry(str):
    try:
        s = float(str)
        return s
    except ValueError:
        pass


def load_movielens_data(data_folder_path):
    """
    The MovieLens dataset is contained at data/ml-100k.zip. This function reads the
    unzipped content of the MovieLens dataset into a numpy array. The file to read in
    is called ```data/ml-100k/u.data``` The description of this dataset is:

    u.data -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of
                  user id | item id | rating | timestamp.
              The time stamps are unix seconds since 1/1/1970 UTC

    Return a numpy array that has size 943x1682, with each item in the matrix (i, j)
    containing the rating user i had for item j. If a user i has no rating for item j,
    you should put 0 for that entry in the matrix.

    Args:
        data_folder_path {str}: Path to MovieLens dataset (given at data/ml-100).
    Returns:
        data {np.ndarray}: Numpy array of size 943x1682, with each item in the array
            containing the rating user i had for item j. If user i did not rate item j,
            the element (i, j) should be 0.
    """
    # This is the path to the file you need to load.
    data_file = os.path.join(data_folder_path, "u.data/ml-100k/u.data")

    data = np.zeros((943, 1682))

    # import zipfile
    # with zipfile.ZipFile(os.path.splitext(data_folder_path)[0]+'.zip',"r") as zip_ref:
    #     zip_ref.extractall(data_folder_path)

    total_num = 0
    with open(data_file) as file:
        for line in csv.reader(file, dialect="excel-tab"):
            # user id | item id | rating  = row, column, rating
            new_line = [read_entry(i) for i in line if read_entry(i) is not None]
            row_num = int(new_line[0]) - 1
            cln_num = int(new_line[1]) - 1
            rating = int(new_line[2])
            data[row_num, cln_num] = rating

            total_num += 1
            # print ("load movielens, new line elements: ", new_line)
            # print ("row_num: ", int(new_line[0]) -1 )
            # print ("cln num", int(new_line[1]) -1)
            # if i==0:
            #     break

    #   data[data == 0] = np.median(data[data != 0])

    return data


# array = load_movielens_data("../data")
