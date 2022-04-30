import matplotlib.pyplot as plt


def show_hist(grey_image_matrix):
    """
    Show histogram of the greyscale image matrix
    :param [] grey_image_matrix: computed greyscale image matrix
    """

    hist = [0 for k in range(256)]
    for row in range(len(grey_image_matrix)):
        for column in range(len(grey_image_matrix[0])):
            hist[grey_image_matrix[row][column]] += 1

    plt.bar([k for k in range(256)], hist, align='center')
    plt.xlim(-1, 256)
    plt.show()


def otsu_threshold(grey_image_matrix) -> float:
    """
    Compute Otsu's threshold
    :param [] grey_image_matrix: matrix of greyscale image
    :return: float
    """

    pixel_number = len(grey_image_matrix) * len(grey_image_matrix[0])
    hist = [0 for k in range(256)]
    for row in range(len(grey_image_matrix)):
        for column in range(len(grey_image_matrix[0])):
            hist[grey_image_matrix[row][column]] += 1

    '''
    We define two classes :
        - First class : from 0 to Threshold
        - Second class : from Threshold to 255
    '''

    # cf https://pastel.archives-ouvertes.fr/tel-01548457/document
    quantity = [0.0 for k in range(256)]  # quantity[i] : quantity of pixels in the first class
    mean = [0.0 for k in range(256)]  # mean[i] : mean of pixels' grey intensity level in the first class
    quantity[0] = hist[0] / pixel_number
    mean[0] = 0
    for i in range(1, 256):
        quantity[i] = quantity[i - 1] + (hist[i] / pixel_number)
        mean[i] = mean[i - 1] + i * (hist[i] / pixel_number)

    threshold = 0
    variance = [0.0 for k in range(256)]  # variance[i] : variance of the first class
    max_variance = 0  # Minimize within-class variance <=> Maximise between-class variance
    for i in range(256):
        if quantity[i] != 0 and quantity[i] != 1:
            variance[i] = (mean[255] * quantity[i] - mean[i]) ** 2 / (quantity[i] * (1 - quantity[i]))
        else:
            variance[i] = 0
        if variance[i] > max_variance:
            max_variance = variance[i]
            threshold = i

    return threshold


def apply_otsu_threshold(grey_image_matrix) -> []:
    """
    Apply Otsu's thresholding to a given greyscale image matrix
    :param [] grey_image_matrix: matrix of a greyscale image
    :return: []
    """

    threshold = otsu_threshold(grey_image_matrix)

    for row in range(len(grey_image_matrix)):
        for column in range(len(grey_image_matrix[0])):
            if grey_image_matrix[row][column] > threshold:
                grey_image_matrix[row][column] = 255
            else:
                grey_image_matrix[row][column] = 0
    return grey_image_matrix
