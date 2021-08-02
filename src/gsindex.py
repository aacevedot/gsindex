from scipy.spatial.distance import cdist
import numpy as np


def geometrical_separability_index(matrix, labels):
    size = len(labels)
    distances = cdist(matrix, matrix)
    positions = distances.argsort(axis=0)
    elements = np.take(labels, positions)
    reference_points = elements[0]
    nearest_neighbours = elements[1]
    gsi = np.sum(reference_points == nearest_neighbours) / size
    return gsi
