import numpy as np


def iou(boxes, clusters):
    """
    Calculates the Intersection over Union (IoU) between N boxes and K clusters.
    :param boxes: numpy array of shape (n, 2) where n is the number of box, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (n, k) where k is the number of clusters
    """
    N = boxes.shape[0]
    K = clusters.shape[0]
    iw = np.minimum(
        np.broadcast_to(boxes[:, np.newaxis, 0], (N, K)),    # (N, 1) -> (N, K)
        np.broadcast_to(clusters[np.newaxis, :, 0], (N, K))  # (1, K) -> (N, K)
    )
    ih = np.minimum(
        np.broadcast_to(boxes[:, np.newaxis, 1], (N, K)),
        np.broadcast_to(clusters[np.newaxis, :, 1], (N, K))
    )
    if np.count_nonzero(iw == 0) > 0 or np.count_nonzero(ih == 0) > 0:
        raise ValueError("Some box has no area")

    intersection = iw * ih   # (N, K)
    boxes_area = np.broadcast_to((boxes[:, np.newaxis, 0] * boxes[:, np.newaxis, 1]), (N, K))
    clusters_area = np.broadcast_to((clusters[np.newaxis, :, 0] * clusters[np.newaxis, :, 1]), (N, K))

    iou_ = intersection / (boxes_area + clusters_area - intersection + 1e-7)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean(np.max(iou(boxes, clusters), axis=1))


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    iter_num = 1
    while True:
        print("Iteration: %d" % iter_num)
        iter_num += 1

        distances = 1 - iou(boxes, clusters)
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            if len(boxes[nearest_clusters == cluster]) == 0:
                print("Cluster %d is zero size" % cluster)
                # to avoid empty cluster
                clusters[cluster] = boxes[np.random.choice(rows, 1, replace=False)]
                continue

            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters
