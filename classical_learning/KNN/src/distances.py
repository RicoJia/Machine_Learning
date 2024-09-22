import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    D = np.empty((0,Y.shape[0]))
    for row_x in X:
        D_row = np.array([])
        for row_y in Y:
            d = np.linalg.norm(row_x-row_y)
            D_row = np.append(D_row, d)
        D = np.vstack((D, D_row))
    return D




def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    D = np.empty((0,Y.shape[0]))
    for row_x in X:
        D_row = np.array([])
        for row_y in Y:
            d = abs(row_x-row_y).sum()
            D_row = np.append(D_row, d)
        D = np.vstack((D, D_row))
    return D


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    D = np.empty((0,Y.shape[0]))
    for row_x in X:
        D_row = np.array([])
        for row_y in Y:
            d = 1- (row_x.dot(row_y))/ np.linalg.norm(row_x)/np.linalg.norm(row_y)
            D_row = np.append(D_row, d)

            # print ("---------------")
            # # print ("row_x: ", row_x)
            # # print ("row_y: ", row_y)
        # print ("d: ", D_row)
        D = np.vstack((D, D_row))
    return D


# x = np.array([[1,0], [-1, 0]])
# y =  np.array([[1,0], [-1, 0], [2,3]])
#
# print (cosine_distances(x, y))
