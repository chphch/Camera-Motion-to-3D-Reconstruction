import numpy as np

from feature import EstimateE_RANSAC
from utils import get_single_right_null_space


def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    
    U, S, VT = np.linalg.svd(E)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    R1 = U @ W @ VT
    R2 = U @ W.T @ VT
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    t = U[:, 2]

    R_set=np.array([R1, R1, R2, R2])
    C_set=np.array([-R1.T @ t, R1.T @ t, -R2.T @ t, R2.T @ t])

    return R_set, C_set



def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """

    mask_valid = np.any(track1 != -1, axis=1) & np.any(track2 != -1, axis=1)
    track1_valid = track1[mask_valid]
    track2_valid = track2[mask_valid]
    valid_indices = np.where(mask_valid)[0]

    n = track1.shape[0]
    X = np.full((n, 3), -1.)
    for valid_index, (x1, y1), (x2, y2) in zip(valid_indices, track1_valid, track2_valid):
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]])
        point3d_homo = get_single_right_null_space(A)
        point3d_eucl = point3d_homo[:3] / point3d_homo[3]
        X[valid_index] = point3d_eucl

    return X


def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The boolean vector indicating the cheirality condition, i.e., the entry 
        is True if the point is in front of both cameras, and False otherwise
    """

    n = X.shape[0]
    mask_available = np.any(X != -1, axis=1)
    indices_available = np.where(mask_available)[0]
    X_valid = X[mask_available]
    
    X_homo = np.insert(X_valid, 3, 1, axis=1)
    z1 = (X_homo @ P1.T)[:, 2]
    z2 = (X_homo @ P2.T)[:, 2]
    valid_index_arr = indices_available[(z1 > 0) & (z2 > 0)]
    valid_index = np.full(n, False)
    valid_index[valid_index_arr] = True

    return valid_index


def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    ransac_n_iter = 1000
    ransac_thr = 0.001

    mask_valid = np.any(track1 != -1, axis=1) & np.any(track2 != -1, axis=1)
    track1_valid = track1[mask_valid]
    track2_valid = track2[mask_valid]

    E, inlier = EstimateE_RANSAC(track1_valid, track2_valid, ransac_n_iter, ransac_thr)
    R_set, C_set = GetCameraPoseFromE(E)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    num_max_valid_index = 0
    R_best = None
    C_best = None
    X_best = None
    for R, C in zip(R_set, C_set):
        P2 = R @ np.hstack([np.eye(3), -C.reshape(-1, 1)])
        X = Triangulation(P1, P2, track1, track2)
        valid_index = EvaluateCheirality(P1, P2, X)
        X[~valid_index] = -1
        num_valid_index = valid_index.sum()
        if num_max_valid_index < num_valid_index:
            num_max_valid_index = num_valid_index
            R_best = R
            C_best = C
            X_best = X
    return R_best, C_best, X_best
