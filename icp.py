import numpy as np


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    return R, t


def transform(arr, R, t):
    '''Transforms arr by R and t'''
    return np.matmul(R, arr.T).T+t


def nearestNeighborDists(points1, points2):
    '''
    Searches nearest neighbor in points2 for points1

    :returns: tuple (nearest neighbor squared distances for points1, indices for nearest neighbor in points2)
    '''
    # pre-allocate to save time
    numPoints = len(points1)
    distances = [0.]*numPoints
    indices = [0]*numPoints

    for iPoint1, point1 in enumerate(points1):
        # search for nearest distance
        dists = np.sum((points2-point1)**2, axis=1)
        iNearest = np.argmin(dists)

        # collect
        indices[iPoint1] = iNearest
        distances[iPoint1] = dists[iNearest]
    
    return distances, indices


def icp(A, B, init_R=None, init_t=None, distance_threshold=0.3, max_iterations=50, tolerance=1e-6):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_R: initial rotation matrix
        init_t: initial translation vector
        distance_threshold: max distance between corresponding points
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        R: final rotation matrix that maps A on to B
        t: final translation vector that maps A on to B
    '''
    src = np.copy(A)
    dst = np.copy(B)

    distance_threshold = distance_threshold**2

    # apply the initial pose estimation
    if init_R is not None and init_t is not None:
        src = transform(src, init_R, init_t)
    
    prev_error = 0

    for _ in range(max_iterations):
        srcs = []
        dsts = []
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearestNeighborDists(src, B)
        for iDist, dist in enumerate(distances):
            if dist < distance_threshold:
                srcs.append(src[iDist])
                dsts.append(dst[indices[iDist]])
        
        if not srcs:
            break

        # compute the transformation between the current source and nearest destination points
        R, t = best_fit_transform(np.vstack(srcs), np.vstack(dsts))

        # update the current source
        src = transform(src, R, t)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        
        prev_error = mean_error

    # calculate final transformation
    R, t = best_fit_transform(A, src)

    return R, t