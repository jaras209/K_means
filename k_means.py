import numpy as np


def k_means_plot(plot, iteration, data, centers, clusters, k):
    import matplotlib.pyplot as plt

    if plot is not True:
        if not plt.gcf().get_axes(): plt.figure(figsize=(4 * 2, 5 * 6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Initialization {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(k))
    if plot is True:
        plt.show()
    else:
        plt.savefig(plot, transparent=True, bbox_inches="tight")


def distance(X: np.ndarray, Y: np.ndarray):
    """
    Returns the matrix `D` where `D[i, j]` is the squared L2 distance of the `i`-th row of X and `j`-th row of Y,
    i.e. `D[i, j] = ||X[i] - Y[j]||^2`.

    :param X: a matrix with data points in rows
    :param Y: a matrix with data points in rows
    :return: a distance matrix of rows of `X` and `Y`
    """
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)

    if Y.ndim == 1:
        Y = np.expand_dims(Y, axis=0)

    D = np.zeros(shape=(X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        diff = x - Y
        norm = np.einsum('ij, ij -> i', diff, diff)
        D[i] = norm

    return D


def k_means(data: np.ndarray, iterations: int, k: int = 10, initialization='random', plot=True,
            random_generator: np.random.RandomState = None, random_seed=42):
    """
    Computes the K-means algorithm with the input `data` using given number of `iterations` and `k` clusters.

    :param data: input data matrix (NumPy array) with data points in the rows
    :param iterations: number of iterations of the K-means algorithm
    :param k: number of clusters
    :param initialization: specify the initialization of the K-means algorithm - it can be either `random` or `kmeans++`
    :param plot: specify whether to plot results and where
    :param random_generator: NumPy random generator for generating initial centers during initialization
    :param random_seed: if `random_generator` is not provided then this seed is used to create one
    :return: a NumPy array of the shape=(data.shape[0]) with integers specifying clusters for each input data point
    """

    assert initialization == 'random' or initialization == 'kmeans++', \
        '`initialization` must be either `random` or `kmeans++`!'

    assert k > 0, '`k` must be strictly greater than 0!'

    random_generator = np.random.RandomState(random_seed) if not random_generator else random_generator

    # Initialize `centers` to be
    # - if args.init == "random", K random data points, using the indices
    #   returned by
    #     `random_generator.choice(len(data), size=args.clusters, replace=False)`
    # - if args.init == "kmeans++", generate the first cluster by
    #     `random_generator.randint(len(data))`
    #   and then iteratively sample the rest of the clusters proportionally to
    #   the square of their distances to their closest cluster using
    #     `random_generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))`
    #   Use the `np.linalg.norm` to measure the distances.
    if initialization == 'random':
        centers_indices = random_generator.choice(data.shape[0], size=k, replace=False)
        centers = data[centers_indices]

    else:
        indices = np.arange(data.shape[0])
        unused_mask = np.ones(shape=data.shape[0], dtype=np.bool)
        centers_indices = [random_generator.randint(data.shape[0])]
        unused_mask[centers_indices] = False
        centers = data[centers_indices]
        for i in range(1, k):
            square_distances = np.min(distance(data[unused_mask], centers), axis=1)
            p = square_distances / np.sum(square_distances)
            centers_indices.append(random_generator.choice(indices[unused_mask], p=p))

            unused_mask[centers_indices] = False
            centers = data[centers_indices]

    if plot:
        k_means_plot(plot, 0, data, centers, clusters=None, k=k)

    clusters = np.zeros(shape=data.shape[0])

    # Run `iterations` of the K-Means algorithm.
    for iteration in range(iterations):
        # Perform a single iteration of the K-Means algorithm, storing
        # zero-based cluster assignment to `clusters`:
        #   1) Compute cluster indices for all data points
        square_distances = distance(data, centers)
        clusters = np.argmin(square_distances, axis=1)

        #   2) Compute new cluster centers using mean of data points assigned to each cluster
        for cluster_index in range(k):
            cluster_mask = clusters == cluster_index
            cluster_data = data[cluster_mask]
            centers[cluster_index] = np.mean(cluster_data, axis=0)

        if plot:
            k_means_plot(plot, 1 + iteration, data, centers, clusters, k=k)

    return clusters
