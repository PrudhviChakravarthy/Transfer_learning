import numpy as np

def cosine_similarity(X, Y):
    a = np.matmul(np.transpose(X), Y)
    b = np.sum(np.multiply(X, X))
    c = np.sum(np.multiply(Y, Y))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def EuclideanDistance(X, Y):
    distance = X - Y
    distance = np.sum(np.multiply(distance, distance))
    distance = np.sqrt(distance)
    return distance