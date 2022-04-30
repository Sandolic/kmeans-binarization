from PIL import Image
import numpy as np
from numpy.linalg import norm
import otsu


# Kmeans class
class Kmeans:
    def __init__(self, n_clusters: int, max_iter=100):
        """
        Kmeans object constructor
        :param int n_clusters: number of clusters - and thus number of centroids
        :param int max_iter: maximum number of iterations inside the "fit" method
        :return Kmeans
        """

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = np.array([])
        self.labels = np.array([])
        self.labels = np.array([])

    def initialize_centroids(self, centroids: np.array):
        """
        Initialize centroids from given ones
        :param np.array centroids: given centroids
        """

        self.centroids = centroids

    def compute_centroids(self, labels: np.array, data: np.array):
        """
        Compute centroids from data and labels
        :param np.array labels: pixels' memberships to different centroids
        :param np.array data: pixels and their RGB values
        """

        centroids = np.zeros((self.n_clusters, data.shape[1]))
        for k in range(self.n_clusters):
            if data[labels == k, :].shape[0] != 0:  # In case no pixels belong to a centroid
                centroids[k, :] = np.mean(data[labels == k, :], axis=0)  # Pixels' colours mean belonging to k-centroid
        return centroids

    def compute_distance(self, data: np.array, centroids: np.array):
        """
        Compute distance between data's pixels and each centroids
        :param np.array data: pixels and their RGB values
        :param np.array centroids: centroids
        """

        distance = np.zeros((data.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(data - centroids[k, :], axis=1)  # Euclidian distance between pixels and k-centroid
            distance[:, k] = np.square(row_norm)
        return distance

    def compute_labels(self, data: np.array, centroids: np.array):
        """
        Compute labels with given centroids
        :param np.array data: pixels and their RGB values
        :param np.array centroids: centroids
        """

        distance = self.compute_distance(data, centroids)
        self.labels = np.argmin(distance, axis=1)  # Closest distance between every pixels and each centroids

    def fit(self, data: np.array):
        """
        K-means method : computing new centroids depending of
        the previous iteration's labels and centroids
        :param np.array data: pixels and their RGB values
        """

        for i in range(self.max_iter):
            old_centroids = self.centroids
            self.compute_labels(data, old_centroids)
            if np.all(old_centroids == self.centroids):  # If the algorithm converges
                break


# Static functions
def compute_local_counter(labels: np.array) -> np.array:
    """
    Compute local counter
    :param np.array labels: pixels' memberships to different centroids
    :return: np.array
    """

    local_counter = np.array([0, 0])
    for k in range(labels.shape[0]):
        slot = labels[k]
        local_counter[slot] += 1
    return local_counter


def compute_local_accumulator(labels: np.array, data: np.array) -> np.array:
    """
    Compute local accumulator
    :param np.array labels: pixels' memberships to different centroids
    :param np.array data: pixels and their RGB values
    :return: np.array
    """

    local_accumulator = np.array([[0, 0, 0], [0, 0, 0]])
    for k in range(labels.shape[0]):
        slot = labels[k]
        local_accumulator[slot, :] += data[k, :]
    return local_accumulator


def compute_global_centroids(counters: np.array, accumulators: np.array) -> np.array:
    """
    Compute global centroids
    :param np.array counters: computed global counters
    :param np.array accumulators: computed global accumulators
    :return: np.array
    """

    global_centroids = np.array([[0, 0, 0], [0, 0, 0]])
    for k in range(2):
        global_centroids[k, :] = np.divide(accumulators[k, :], counters[k])
    return global_centroids


def binarization(file_path: str):
    """
    Binarize image from given file path
    :param string file_path: given file path
    """

    '''
    Instead of finding centroids in the image itself - which would bias the results because
    of the image's features - we consider a 3-dimensional space where each axes are the
    Red, Green, Blue, of each pixel. Thus, the process is only dependant from the image's colours.
    
    Furthermore, to be sure not to have biased results based on the global colour features,
    we will divide the image into chunks, and process the K-means method - which uses "centroids"
    that are explained below - for each of them. Then, we combine the local results into 
    a global one using "counters" and "accumulators".
    
    
    Centroid : Virtual point from which data can belong to. With more than one centroid, data
    is divided into "clusters", which can help processing it.
      - In our case : we want to binarize an image, to we will use two centroids (either local or global)
      to regroup pixels into.
    
    Counter : number of pixels that belong to each centroids.
      - In our case : [a, b] where a the number of pixels belonging to the first centroid, and
      b the number of pixels belonging to the second centroid.
    
    Accumulator : Sum of all pixels' R, G, B components belonging to each centroids.
      - In our case : [[R1, G1, B1], [R2, B2, G2]] where X1 is the sum of all pixels' X component
      belonging to the first centroid, and X2 the sum of all pixels' X component belonging to the
      second centroid.
    '''

    # --- Initialization ---
    image_matrix = np.array(Image.open(file_path).convert("RGB"))
    data = image_matrix.reshape(image_matrix.shape[0] * image_matrix.shape[1], image_matrix.shape[2])
    chunks_number = 512
    sub_data = np.array_split(data, chunks_number)
    km = Kmeans(n_clusters=2)

    # --- Local k-means method ---
    global_centroids = np.array([[0, 0, 0], [255, 255, 255]])
    old_global_centroids = np.array([[0, 0, 0], [0, 0, 0]])
    while global_centroids.any() != old_global_centroids.any():
        old_global_centroids = global_centroids
        local_counters = np.array([0, 0])
        local_accumulators = np.array([[0, 0, 0], [0, 0, 0]])
        for k in range(chunks_number):
            km.initialize_centroids(global_centroids)
            km.fit(sub_data[k])
            local_counters += compute_local_counter(km.labels)
            local_accumulators += compute_local_accumulator(km.labels, sub_data[k])
        global_centroids = compute_global_centroids(local_counters, local_accumulators)

    # --- Extracting binarized image ---
    km.initialize_centroids(global_centroids)
    km.compute_labels(data, km.centroids)
    kmeans_matrix = km.centroids[km.labels]  # Get pixels with their binary colour
    kmeans_matrix = np.clip(kmeans_matrix.astype("uint8"), 0, 255)
    kmeans_matrix = kmeans_matrix.reshape(image_matrix.shape[0], image_matrix.shape[1], image_matrix.shape[2])
    binary_image_matrix = otsu.apply_otsu_threshold(np.array(Image.fromarray(kmeans_matrix).convert("L")))
    Image.fromarray(binary_image_matrix).save(file_path + "_binary.png")
