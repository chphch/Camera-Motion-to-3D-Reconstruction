import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils import get_single_right_null_space


def extractSIFT(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    loc = np.array([k.pt for k in kp])
    return loc, des


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

    filter_ratio = 0.5
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(des2)
    neigh_dist_1_img2, neigh_ind_1_img2 = neigh.kneighbors(des1, return_distance=True) # (n1,?)
    neigh.fit(des1)
    neigh_dist_2_img1, neigh_ind_2_img1 = neigh.kneighbors(des2, return_distance=True) # (n2,?)

    filtered_bool_1_1to2 = neigh_dist_1_img2[:, 0] < filter_ratio * neigh_dist_1_img2[:, 1]
    filtered_bool_2_2to1 = neigh_dist_2_img1[:, 0] < filter_ratio * neigh_dist_2_img1[:, 1]
    filtered_ind_2_1to2 = neigh_ind_1_img2[filtered_bool_1_1to2, 0]
    filtered_bool_2_bi = np.in1d(np.arange(loc2.shape[0]), filtered_ind_2_1to2) & filtered_bool_2_2to1
    ind2 = np.where(filtered_bool_2_bi)[0]
    ind1 = neigh_ind_2_img1[ind2, 0]

    x1 = loc1[ind1]
    x2 = loc2[ind2]
    return x1, x2, ind1


def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """

    x = x1[:, [0]]
    y = x1[:, [1]]
    x_ = x2[:, [0]]
    y_ = x2[:, [1]]
    n = x1.shape[0]
    A = np.hstack([x_ * x, x_ * y, x_, y_ * x, y_ * y, y_, x, y, np.ones((n, 1))])
    f = get_single_right_null_space(A)
    E_raw = f.reshape(3, 3)
    U, S, VT = np.linalg.svd(E_raw)
    S_E = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]])
    E = U @ S_E @ VT
    return E


def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix

    inlier : ndarray of shape (k,)
        The inlier indices
    """

    n = x1.shape[0]
    inlier_best = np.array([])
    E_best = None
    x1_homo = np.insert(x1, 2, 1, axis=1)
    x2_homo = np.insert(x2, 2, 1, axis=1)
    for _ in range(ransac_n_iter):
        indices = np.random.choice(n, 8, replace=False)
        E = EstimateE(x1[indices], x2[indices])
        xEx = (x2_homo * (x1_homo @ E.T)).sum(axis=1)
        inlier = np.where(np.abs(xEx) < ransac_thr)[0]
        if inlier_best.shape[0] < inlier.shape[0]:
            E_best = E
            inlier_best = inlier
    return E_best, inlier_best


def BuildFeatureTrack(Im, K):
    """
    Build feature track
    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters
    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """

    ransac_n_iter = 500
    ransac_thr = 0.01
    N = Im.shape[0]
    K_inv = np.linalg.inv(K)
    SIFTs = [extractSIFT(im) for im in Im]  # [(loc, des)]
    
    track = np.empty((N, 0, 2))
    for i in range(N - 1):
        loc_i, des_i = SIFTs[i]
        F = loc_i.shape[0]
        track_i = np.full((N, F, 2), -1.)
        matched_indices = set()

        for j in range(i+1, N):
            loc_j, des_j = SIFTs[j]
            x1, x2, ind1 = MatchSIFT(loc_i, des_i, loc_j, des_j)
            x1_homo = np.insert(x1, 2, 1, axis=1)
            x2_homo = np.insert(x2, 2, 1, axis=1)
            x1_norm_eucl = (x1_homo @ K_inv.T)[:, :2]
            x2_norm_eucl = (x2_homo @ K_inv.T)[:, :2]
            E, inlier = EstimateE_RANSAC(x1_norm_eucl, x2_norm_eucl, ransac_n_iter, ransac_thr)
            inlier_ind_in_track = ind1[inlier]
            track_i[i, inlier_ind_in_track] = x1_norm_eucl[inlier]
            track_i[j, inlier_ind_in_track] = x2_norm_eucl[inlier]
            matched_indices = matched_indices | set(inlier_ind_in_track)
            # visualize_epipolar_lines(Im, i, j, x1_homo, x2_homo, K, E, inlier)

        track_i = track_i[:, np.array(list(matched_indices), dtype=int)]
        track = np.concatenate([track, track_i], axis=1)
        visualization(Im, i, j, track, E, K)
    return track


def visualize_epipolar_lines(Im, i, j, x1_homo, x2_homo, K, E, inlier):
    im1 = Im[i].copy()
    im2 = Im[j].copy()
    x1_inlier_homo = x1_homo[inlier]
    x2_inlier_homo = x2_homo[inlier]
    F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
    l2 = x1_inlier_homo @ F.T
    l1 = x2_inlier_homo @ F
    x1_inlier_eucl = (x1_inlier_homo / x1_inlier_homo[:, [2]])[:, :2]
    x2_inlier_eucl = (x2_inlier_homo / x2_inlier_homo[:, [2]])[:, :2]

    for (xp1_x, xp1_y), (xp2_x, xp2_y), lp1, lp2 in zip(x1_inlier_eucl, x2_inlier_eucl, l1, l2):
        color = np.random.randint(256, size=3, dtype=int).tolist()
        cv2.circle(im1, (int(xp1_x), int(xp1_y)), radius=3, color=color, thickness=cv2.FILLED)
        cv2.circle(im2, (int(xp2_x), int(xp2_y)), radius=3, color=color, thickness=cv2.FILLED)
        cv2.line(im1, *get_points_from_line(im1, lp1), color=color, thickness=2)
        cv2.line(im2, *get_points_from_line(im2, lp2), color=color, thickness=2)
    im = np.hstack([im1, im2])
    cv2.imwrite(f'test{i}_{j}.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def get_points_from_line(im, line_homo):
    if line_homo[0] != 0 and line_homo[1] != 0:
        p1 = (int(-line_homo[2] / line_homo[0]), 0)
        p2 = (0, int(-line_homo[2] / line_homo[1]))
    elif line_homo[0] != 0 and line_homo[1] == 0:
        p1 = (int(-line_homo[2] / line_homo[0]), 0)
        p2 = (int(-line_homo[2] / line_homo[0]), im.shape[0] - 1)
    elif line_homo[0] == 0 and line_homo[1] != 0:
        p1 = (im.shape[1] - 1, int(-line_homo[2] / line_homo[1]))
        p2 = (0, int(-line_homo[2] / line_homo[1]))
    else:
        p1 = p2 = (0, 0)
    return p1, p2



def visualization(Im, i, j, track, E, K):
    # Visualize epipolar lines
    idx1 = i
    idx2 = j
    img_merged = np.concatenate((Im[idx1], Im[idx2]), axis=1)
    shifted = Im[idx1].shape[1]
    for f in range(track.shape[1]):
        p1 = track[idx1, f, :]
        p2 = track[idx2, f, :]
        if np.any(p1 != -1) and np.any(p2 != -1):
            p1 = np.concatenate((p1, [1]))
            p2 = np.concatenate((p2, [1]))
            # unnormalize
            p1 = K @ p1
            p2 = K @ p2
            F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
            # ax+by+c = 0 -> put 0 & img_width to x to calculate start point & end point of the line
            line2 = F @ p1
            line2_start = (0, -line2[2]/line2[1])
            line2_end = ( Im[idx2].shape[1], (-line2[0]*Im[idx2].shape[1] - line2[2]) / line2[1] )
            line1 = F.T @ p2
            line1_start = (0, -line1[2]/line1[1])
            line1_end = ( Im[idx1].shape[1], (-line1[0]*Im[idx1].shape[1] - line1[2]) / line1[1] )
            color = np.random.randint(0, 255, 3).tolist()
            # Draw epipolar lines and the matched feature points on them
            cv2.line(img_merged,
                (int(line1_start[0]), int(line1_start[1])),
                (int(line1_end[0]), int(line1_end[1])),
                color , 2)
            cv2.line(img_merged,
                    (int(line2_start[0] + shifted), int(line2_start[1])),
                    (int(line2_end[0] + shifted), int(line2_end[1])),
                    color , 2)
            cv2.circle(img_merged, (int(p1[0]), int(p1[1])), radius=5, color=color, thickness=-1)
            cv2.circle(img_merged, (int(p2[0] + shifted), int(p2[1])), radius=5, color=color, thickness=-1)
    cv2.imwrite(f"output/tmp{idx1}{idx2}.jpg", img_merged)