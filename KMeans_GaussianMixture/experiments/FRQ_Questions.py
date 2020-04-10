import numpy as np
from src import load_mnist, adjusted_mutual_info
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy.stats import mode


def FRQ_3_a():
    #load images that are: image:  (60000, 28, 28), labels: labels:  [5 0 4 ... 5 6 8],
    images, labels = load_mnist(dataset= 'testing', path="/home/ricojia/win2020-hw6-clustering-RicoJia/data")
    digit_list, image_count = np.unique(labels, return_counts=True)
    n = np.min(image_count)

    # #flatten images
    images = images.reshape(10000, 784)

    # storing images as [[first digit imgs], [second digit imgs], ...]
    all_training_imgs = np.empty((0, 784))
    all_training_labels = np.empty(0)
    for i_num in range(10):
        digit_imgs = []
        digit_labels = []
        for i in range(10000):
            if labels[i]==i_num and len(digit_imgs)<n:
                digit_imgs.append(images[i, :])
                digit_labels.append(i_num)
        digit_imgs = np.array(digit_imgs)
        digit_labels = np.array(digit_labels)
        all_training_imgs = np.append(all_training_imgs, digit_imgs, axis=0)
        all_training_labels = np.append(all_training_labels, digit_labels)

    # diag covariance
    # diag_gmm = GaussianMixture(n_components=10, covariance_type='diag')
    # diag_gmm.fit(all_training_imgs)
    # model_labels = diag_gmm.fit_predict(all_training_imgs)

    # sphe_gmm = GaussianMixture(n_components=10, covariance_type='spherical')
    # sphe_gmm.fit(all_training_imgs)
    # model_labels = sphe_gmm.fit_predict(all_training_imgs)
    # MI_Score = adjusted_mutual_info(model_labels, all_training_labels)
    # print("spherical GMM MI score: ", MI_Score)

    kmeans= KMeans(n_clusters=10, random_state = 0)
    kmeans.fit(all_training_imgs)
    model_labels = kmeans.predict(all_training_imgs)
    MI_Score = adjusted_mutual_info(model_labels, all_training_labels)
    print("kmeans GMM MI score: ", MI_Score)



def get_actual_labels(real_labels, cluster_pred, cluster_size, cluster_num):
    """
    Return the labels of each digit, by:
        1. identifying each cluster
        2. finding the mode of the true labels of each cluster, and use it as the label of each cluster
        3. Relabel each point with the label of its cluster
    Args:
        1. real_labels: [n_samples, 1] array of the real labels of each feature vector
        2. clster_pred: [n_samples, 1] array of cluster id of each point, as the output of the model predict function
        3. cluster_size: size of the cluster
        4. cluster_num: number of clusters.

    Return:
        label of the cluster of each feature vector.
    """

    #idenftify each cluster's label, and store it in a dictionary
    cluster_labels = {}
    for i_cluster in range(cluster_num):
        cluster_label_list = np.array( [real_labels[i] for i in range(real_labels.shape[0]) if cluster_pred[i] == i_cluster] )
        cluster_label_mode = mode(cluster_label_list)[0][0]
        cluster_labels[i_cluster] = cluster_label_mode

    new_labels = np.copy(cluster_pred)
    for i_feature in range(new_labels.shape[0]):
        new_labels[i_feature] = cluster_labels[cluster_pred[i_feature]]
    return new_labels

def FRQ_3_b():
    #load images that are: image:  (60000, 28, 28), labels: labels:  [5 0 4 ... 5 6 8],
    images, labels = load_mnist(dataset= 'testing', path="/home/ricojia/win2020-hw6-clustering-RicoJia/data")
    digit_list, image_count = np.unique(labels, return_counts=True)
    n = np.min(image_count)

    # #flatten images
    images = images.reshape(10000, 784)

    # storing images as [[first digit imgs], [second digit imgs], ...]
    all_training_imgs = np.empty((0, 784))
    all_training_labels = np.empty(0)
    for i_num in range(10):
        digit_imgs = []
        digit_labels = []
        for i in range(10000):
            if labels[i]==i_num and len(digit_imgs)<n:
                digit_imgs.append(images[i, :])
                digit_labels.append(i_num)
        digit_imgs = np.array(digit_imgs)
        digit_labels = np.array(digit_labels)
        all_training_imgs = np.append(all_training_imgs, digit_imgs, axis=0)
        all_training_labels = np.append(all_training_labels, digit_labels)

    # # diag covariance
    # diag_gmm = GaussianMixture(n_components=10, covariance_type='diag')
    # diag_gmm.fit(all_training_imgs)
    # model_labels = diag_gmm.fit_predict(all_training_imgs)
    # model_labels = get_actual_labels(all_training_labels, model_labels, cluster_num=10, cluster_size=892)
    # print("model labels for the first 100 features: ", model_labels[: 100])
    # score = accuracy_score(all_training_labels, model_labels)
    # print("spherical diag score: ", score)

    sphe_gmm = GaussianMixture(n_components=10, covariance_type='spherical')
    sphe_gmm.fit(all_training_imgs)
    model_labels = sphe_gmm.predict(all_training_imgs)
    model_labels = get_actual_labels(all_training_labels, model_labels, cluster_num=10, cluster_size=892)
    print("model labels for the first 100 features: ", model_labels[: 100])
    score = accuracy_score(all_training_labels, model_labels)
    print("spherical GMM score: ", score)

    # kmeans= KMeans(n_clusters=10, random_state = 0)
    # kmeans.fit(all_training_imgs)
    # model_labels = kmeans.predict(all_training_imgs)
    # model_labels = get_actual_labels(all_training_labels, model_labels, cluster_num=10, cluster_size=892)
    # print("model labels for the first 100 features: ", model_labels[: 100])
    # score = accuracy_score(all_training_labels, model_labels)
    # print("Kmeans GMM score: ", score)

def find_cluster_mean(imgs, real_labels, cluster_pred, cluster_size, cluster_num):
    """
    Find each cluster, their label, and their mean for each pixel
    Args:
        0. imgs: [10000, 786] imgs
        1. real_labels: [n_samples, 1] array of the real labels of each feature vector
        2. clster_pred: [n_samples, 1] array of cluster id of each point, as the output of the model predict function
        3. cluster_size: size of the cluster
        4. cluster_num: number of clusters.

    Return:
        Dictionary with {'real_label_of_cluster', mean}
    """

    #idenftify each cluster's label, and store it in a dictionary
    cluster_label_mean = {}
    #initialize each item with a randomly selected img, because a cluster mode might be the same as another cluster's mode, which will miss a digit
    for i_digit in range(10):
        first_img_index = real_labels.tolist().index(i_digit)
        cluster_label_mean[i_digit] = imgs[first_img_index]

    for i_cluster in range(cluster_num):
        cluster_label_list = np.array( [real_labels[i] for i in range(real_labels.shape[0]) if cluster_pred[i] == i_cluster] )
        cluster_label_mode = mode(cluster_label_list)[0][0]
        print("most common label: ", cluster_label_mode)

        #TODO
        cluster_imgs = np.array([imgs[i] for i in range(cluster_pred.shape[0]) if cluster_pred[i]==i_cluster])
        cluster_mean = np.mean(cluster_imgs, axis=0)
        cluster_label_mean [cluster_label_mode] = cluster_mean

    return cluster_label_mean



def find_cluster_mean_closest(imgs, real_labels, cluster_pred, cluster_size, cluster_num):
    """
    Find each cluster, their label, and their mean for each pixel
    Args:
        0. imgs: [10000, 786] imgs
        1. real_labels: [n_samples, 1] array of the real labels of each feature vector
        2. clster_pred: [n_samples, 1] array of cluster id of each point, as the output of the model predict function
        3. cluster_size: size of the cluster
        4. cluster_num: number of clusters.

    Return:
        Dictionary with {'real_label_of_cluster', mean}
    """

    #idenftify each cluster's label, and store it in a dictionary
    closest_imgs = {}
    #initialize each item with a randomly selected img, because a cluster mode might be the same as another cluster's mode, which will miss a digit
    for i_digit in range(10):
        first_img_index = real_labels.tolist().index(i_digit)
        closest_imgs[i_digit] = imgs[first_img_index]

    for i_cluster in range(cluster_num):
        cluster_label_list = np.array( [real_labels[i] for i in range(real_labels.shape[0]) if cluster_pred[i] == i_cluster] )
        cluster_label_mode = mode(cluster_label_list)[0][0]
        print("most common label: ", cluster_label_mode)

        cluster_imgs = np.array([imgs[i] for i in range(cluster_pred.shape[0]) if cluster_pred[i]==i_cluster])
        cluster_mean = np.mean(cluster_imgs, axis=0)

        #find euclidean distance of each point
        cluster_img_dists_to_mean = np.array([np.linalg.norm(imgs[j] - cluster_mean) for j in range(cluster_imgs.shape[0]) ])
        closest_img = cluster_imgs[np.argmin(cluster_img_dists_to_mean)]
        closest_imgs[cluster_label_mode] = closest_img

    return closest_imgs

def FRQ_3_c():
    #use spherical GMM
    #load images that are: image:  (60000, 28, 28), labels: labels:  [5 0 4 ... 5 6 8],
    images, labels = load_mnist(dataset= 'testing', path="/home/ricojia/win2020-hw6-clustering-RicoJia/data")
    digit_list, image_count = np.unique(labels, return_counts=True)
    n = np.min(image_count)

    # #flatten images
    images = images.reshape(10000, 784)

    # storing images as [[first digit imgs], [second digit imgs], ...]
    all_training_imgs = np.empty((0, 784))
    all_training_labels = np.empty(0)
    for i_num in range(10):
        digit_imgs = []
        digit_labels = []
        for i in range(10000):
            if labels[i]==i_num and len(digit_imgs)<n:
                digit_imgs.append(images[i, :])
                digit_labels.append(i_num)
        digit_imgs = np.array(digit_imgs)
        digit_labels = np.array(digit_labels)
        all_training_imgs = np.append(all_training_imgs, digit_imgs, axis=0)
        all_training_labels = np.append(all_training_labels, digit_labels)

    sphe_gmm = GaussianMixture(n_components=10, covariance_type='spherical')
    sphe_gmm.fit(all_training_imgs)
    model_labels = sphe_gmm.predict(all_training_imgs)

    # test
    # model_labels = all_training_labels

    cluster_means = find_cluster_mean(imgs=all_training_imgs, real_labels=all_training_labels, cluster_pred=model_labels, cluster_num=10, cluster_size=892)

    ax = []
    fig = plt.figure(figsize=(8, 8))
    rows = 5
    columns = 2
    i = 0
    for flattened_img_key in cluster_means:
        img = cluster_means[flattened_img_key].reshape(28,28)
        ax.append(fig.add_subplot(rows, columns, flattened_img_key+1))  #TODO
        ax[-1].set_title("digit: "+ str(flattened_img_key))  # set title
        plt.imshow(img, alpha=0.25)
        i+=1

    closet_imgs = find_cluster_mean_closest(imgs=all_training_imgs, real_labels=all_training_labels, cluster_pred=model_labels, cluster_num=10, cluster_size=892)
    ax2 = []
    fig = plt.figure(figsize=(8, 8))
    rows = 5
    columns = 2
    i = 0
    for flattened_img_key in closet_imgs:
        img = closet_imgs[flattened_img_key].reshape(28,28)
        ax2.append(fig.add_subplot(rows, columns, flattened_img_key+1))  #TODO
        ax2[-1].set_title("digit: "+ str(flattened_img_key))  # set title
        plt.imshow(img, alpha=0.25)
        i+=1

    plt.show()



def find_all_imgs(images, labels, digit):
    """
    finding all images of  digit from images
    Return:
        [m, 786]
    """
    imgs = []
    for i_image in range(images.shape[0]):
        if labels[i_image] == digit:
            imgs.append(images[i_image, :])
    return np.array(imgs)

def FRQ_bonus():
    #load images that are: image:  (60000, 28, 28), labels: labels:  [5 0 4 ... 5 6 8],
    images, labels = load_mnist(dataset= 'testing', path="/home/ricojia/win2020-hw6-clustering-RicoJia/data")

    # choose digit0 images
    all_training_imgs = find_all_imgs(images=images.reshape(10000,784), labels=labels, digit=0)

    component_nums = [1,4,10,20]
    imgs = []
    for i in range(4):
        sphe_gmm = GaussianMixture(n_components=component_nums[i], covariance_type='spherical')
        sphe_gmm.fit(all_training_imgs)
        new_sample_imgs,labels = sphe_gmm.sample(n_samples= 5)

        #test
        # new_sample_imgs = np.ones((5, 784))
        imgs.append(new_sample_imgs)

    imgs = np.array(imgs)


    for cluster_index in range(imgs.shape[0]):
        cluster_imgs = imgs[cluster_index, :, :]
        ax = []
        fig = plt.figure(figsize=(8,8))
        rows = 3
        columns = 2
        i = 0
        for pic_index in range(5):
            img = cluster_imgs[pic_index, :].reshape(28,28)
            ax.append(fig.add_subplot(rows, columns, pic_index+1))  #TODO
            ax[-1].set_title("sample "+ str(pic_index))  # set title
            ax[-1].axis("off")
            plt.imshow(img, alpha=0.25)
            i+=1
        fig.suptitle("cluster number: "+ str(component_nums[cluster_index]))
        plt.show()


FRQ_bonus()

