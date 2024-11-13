import matplotlib.pyplot as plt
import numpy as np
from load_movielens import load_movielens_data


def collaborative_filtering(
    input_array, n_neighbors, distance_measure="euclidean", aggregator="mode"
):

    impute_array = np.zeros(input_array.shape[1])
    cp_input_array = np.copy(input_array)
    cp2_input_array = np.copy(input_array)
    for cln_num in range(cp_input_array.shape[1]):
        sorted_cln = cp2_input_array[:, cln_num]
        sorted_cln.sort()
        first_non_zero_index = (sorted_cln == 0).sum()
        k_neighbors = sorted_cln[
            first_non_zero_index : first_non_zero_index + n_neighbors
        ]
        k_neighbors_ls = k_neighbors.tolist()
        if aggregator == "mode":
            impute_array[cln_num] = max(set(k_neighbors_ls), key=k_neighbors_ls.count)

        elif aggregator == "mean":
            impute_array[cln_num] = k_neighbors.sum() / k_neighbors.shape[0]
        elif aggregator == "median":
            impute_array[cln_num] = k_neighbors_ls[int(k_neighbors.shape[0] / 2)]

        for row_num in range(input_array.shape[0]):
            if cp_input_array[row_num][cln_num] == 0:
                cp_input_array[row_num][cln_num] = impute_array[cln_num]

    return cp_input_array


def sub_0(new_array, original_array, N_nominal):
    N_min = np.min(np.count_nonzero(original_array, axis=1))
    N = N_min if N_min < N_nominal else N_nominal

    subed_new_array = np.copy(new_array)
    for row_num in range(new_array.shape[0]):
        nonzero_indices = np.array(np.where(original_array[row_num] != 0)[0])
        np.random.shuffle(nonzero_indices)
        to_delete_indices = nonzero_indices[:N]

        for n in to_delete_indices:
            subed_new_array[row_num, n] = 0
    return subed_new_array, N * new_array.shape[0]


def get_MSE(estimates, targets, imputed_num):
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is
    estimated. The formula is:

    MSE = (1 / n) * \sum_{i=1}^{n} (Y_i - Yhat_i)^2

    Implement this formula here, using numpy and return the computed MSE

    https://en.wikipedia.org/wiki/Mean_squared_error
    Input: 2 numpy arrays
    Output: Mean Square Error

    this does not work if you have a matrix!
    """
    n = estimates.shape[0]
    m = estimates.shape[1]
    mse = np.sqrt(((targets - estimates) ** 2).sum() / imputed_num)

    # mse = np.sqrt(1.0/m*1.0/n*(((targets-estimates)*(targets-estimates)).sum()))
    return mse


def collab_MSE(N, K, D, A, new_array, old_array):

    subed_new_array, imputed_num = sub_0(new_array, old_array, N)
    filtered_array = collaborative_filtering(subed_new_array, K, D, A)
    MSE = get_MSE(filtered_array, old_array, imputed_num)

    return MSE


def main():
    # load data
    array = load_movielens_data("../data")
    new_array = np.copy(array)

    # fill in all zeros with median of the same row
    for row_num in range(new_array.shape[0]):
        median = np.median(np.array([i for i in new_array[row_num, :] if i != 0]))
        new_array[row_num, :] = np.where(
            new_array[row_num, :] == 0, median, new_array[row_num, :]
        )

    D_arr = ["euclidean", "manhattan", "cosine"]
    A_arr = ["mode", "mean", "median"]
    MSE_arr = np.array([])

    # question 7
    # total_val = (array.shape[0]-1)*(array.shape[0])/2
    # similar_array = np.zeros(int(total_val))
    # index = 0
    # print (similar_array.shape)
    # for user_row_index in range(array.shape[0]-1):
    #     for another_user_index in range(user_row_index+1, array.shape[0]):
    #         common_total = (np.logical_and((array[user_row_index] !=0 ), (array[another_user_index]!=0))).sum()
    #         similar_array[index] = common_total
    #         index+=1
    #
    # # mean= similar_array.sum()/((similar_array!=0).sum())
    # median = np.median(similar_array[similar_array!=0])
    #
    # # print("Q7 - mean: ", mean)
    # print("Q7 - mean: ", median)

    # question 8
    # N_arr = [5, 10, 20, 40]
    # K = 3
    # D = D_arr[0]
    # A = A_arr[1]
    # for N in N_arr:
    #     MSE = collab_MSE(N, K, D, A, new_array, array)
    #     print("MSE: ", MSE)
    #     MSE_arr = np.append(MSE_arr, MSE)
    # plt.plot(N_arr, MSE_arr)
    # plt.title("MSE vs Changes In N")
    # plt.xlabel("N")
    # plt.ylabel("MSE")
    # plt.savefig("Question8.png")

    # # question 9
    # N = 1
    # K = 3
    # A = 'mean'
    # for D in D_arr:
    #     MSE = collab_MSE(N, K, D, A, new_array, array)
    #     print("MSE: ", MSE)
    #     MSE_arr = np.append(MSE_arr, MSE)
    # plt.plot(D_arr, MSE_arr)
    # # plt.plot(D_arr, np.around(MSE_arr, decimals=3))
    # plt.title("MSE vs Changes In Distance Measure")
    # plt.xlabel("Distance Measure")
    # plt.ylabel("MSE")
    # plt.savefig("Question9.png")

    # #question 10
    # N = 1
    # D = 'manhattan'
    # A = 'mean'
    # K_arr = [1, 3, 7, 11, 15, 31]
    # for K in K_arr:
    #     MSE = collab_MSE(N, K, D, A, new_array, array)
    #     print("MSE: ", MSE)
    #     MSE_arr = np.append(MSE_arr, MSE)
    # plt.plot(K_arr, MSE_arr)
    # plt.title("MSE vs Changes In K")
    # plt.xlabel("K")
    # plt.ylabel("MSE")
    # plt.savefig("Question10.png")

    # question 11
    N = 1
    D = "manhattan"
    K = 31
    A_arr = ["mean", "mode", "median"]
    for A in A_arr:
        MSE = collab_MSE(N, K, D, A, new_array, array)
        print("MSE: ", MSE)
        MSE_arr = np.append(MSE_arr, MSE)
    plt.plot(A_arr, MSE_arr)
    plt.title("MSE vs Changes In Aggregator")
    plt.xlabel("Aggregator")
    plt.ylabel("MSE")
    plt.savefig("Question11.png")


main()
