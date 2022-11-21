# import required libraries
import numpy as np
from numpy.linalg import norm

def similarity(A,B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine

if __name__ == "__main__":
    # define two lists or array
    A = np.array([2,1,2,3,2,9])
    B = np.array([3,4,2,4,5,5])
    score = similarity(A,B)
    print(f"Score :: {score}")

