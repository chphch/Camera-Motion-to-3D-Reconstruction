import numpy as np

from utils import Rotation2Quaternion, get_single_right_null_space
from utils import Quaternion2Rotation


def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """
    indices = np.random.choice(X.shape[0], 6)
    X_sample = X[indices]
    x_sample = x[indices]
    A = np.vstack([[
        [Xx, Xy, Xz, 1, 0, 0, 0, 0, -xx * Xx, -xx * Xy, -xx * Xz, -xx],
        [0, 0, 0, 0, Xx, Xy, Xz, 1, -xy * Xx, -xy * Xy, -xy * Xz, -xy],
    ] for (Xx, Xy, Xz), (xx, xy) in zip(X_sample, x_sample)])
    f = get_single_right_null_space(A)
    P = f.reshape(3, 4)
    R = P[:, :3]
    U, S, VT = np.linalg.svd(R)
    R = U @ VT
    t = 1 / S[0] * P[:, 3]
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    C = -R.T @ t
    return R, C


def PnP_RANSAC(X, x, ransac_n_iter, ransac_thr):
    """
    Estimate pose using PnP with RANSAC

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    inlier : ndarray of shape (n,)
        The indicator of inliers, i.e., the entry is True if the point is a inlier,
        and False otherwise
    """
    
    n = X.shape[0]
    X_homo = np.insert(X, 3, 1, axis=1)
    R_best = None
    C_best = None
    mask_best = np.array([], dtype=bool)
    for _ in range(ransac_n_iter):
        R, C = PnP(X, x)
        P = R @ np.hstack([np.eye(3), -C.reshape(-1, 1)])
        X_proj = X_homo @ P.T
        x_eucl = X_proj[:, :2] / X_proj[:, [2]]  # X_proj = x_homo
        mask_valid = X_proj[:, 2] > 0
        mask_inlier = np.linalg.norm(x_eucl - x, axis=1) < ransac_thr
        mask = mask_valid & mask_inlier
        if np.sum(mask_best) < np.sum(mask):
            R_best = R
            C_best = C
            mask_best = mask
    
    inlier = np.full(n, False)
    inlier[mask_best] = True
    
    return R_best, C_best, inlier


def ComputePoseJacobian(p, X):
    """
    Compute the pose Jacobian

    Parameters
    ----------
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion
    X : ndarray of shape (3,)
        3D point

    Returns
    -------
    dfdp : ndarray of shape (2, 7)
        The pose Jacobian
    """
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)
    x = R @ (X - C)

    u = x[0]
    v = x[1]
    w = x[2]
    du_dc = -R[0,:]
    dv_dc = -R[1,:]
    dw_dc = -R[2,:]
    # df_dc is in shape (2, 3)
    df_dc = np.stack([
        (w * du_dc - u * dw_dc) / (w**2),
        (w * dv_dc - v * dw_dc) / (w**2)
    ], axis=0)

    # du_dR = np.concatenate([X-C, np.zeros(3), X-C])
    # dv_dR = np.concatenate([np.zeros(3), X-C, X-C])
    # dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X-C])
    du_dR = np.concatenate([X-C, np.zeros(3), np.zeros(3)])
    dv_dR = np.concatenate([np.zeros(3), X-C, np.zeros(3)])
    dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X-C])
    # df_dR is in shape (2, 9)
    df_dR = np.stack([
        (w * du_dR - u * dw_dR) / (w**2),
        (w * dv_dR - v * dw_dR) / (w**2)
    ], axis=0)


    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    # dR_dq is in shape (9, 4)
    dR_dq = np.asarray([
        [0, 0, -4*qy, -4*qz],
        [-2*qz, 2*qy, 2*qx, -2*qw],
        [2*qy, 2*qz, 2*qw, 2*qx],
        [2*qz, 2*qy, 2*qx, 2*qw],
        [0, -4*qx, 0, -4*qz],
        [-2*qx, -2*qw, 2*qz, 2*qy],
        [-2*qy, 2*qz, -2*qw, 2*qx],
        [2*qx, 2*qw, 2*qz, 2*qy],
        [0, -4*qx, -4*qy, 0],
    ])

    dfdp = np.hstack([df_dc, df_dR @ dR_dq])

    return dfdp


def PnP_nl(R, C, X, x):
    """
    Update the pose using the pose Jacobian

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix refined by PnP
    c : ndarray of shape (3,)
        Camera center refined by PnP
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R_refined : ndarray of shape (3, 3)
        The rotation matrix refined by nonlinear optimization
    C_refined : ndarray of shape (3,)
        The camera center refined by nonlinear optimization
    """
    n = X.shape[0]
    q = Rotation2Quaternion(R)

    p = np.concatenate([C, q])
    n_iters = 20
    lamb = 1
    error = np.empty((n_iters,))
    for i in range(n_iters):
        R_i = Quaternion2Rotation(p[3:])
        C_i = p[:3]

        proj = (X - C_i[np.newaxis,:]) @ R_i.T
        proj = proj[:,:2] / proj[:,2,np.newaxis]

        H = np.zeros((7,7))
        J = np.zeros(7)
        for j in range(n):
            dfdp = ComputePoseJacobian(p, X[j,:])
            H = H + dfdp.T @ dfdp
            J = J + dfdp.T @ (x[j,:] - proj[j,:])
        
        delta_p = np.linalg.inv(H + lamb*np.eye(7)) @ J
        p += delta_p
        p[3:] /= np.linalg.norm(p[3:])

        error[i] = np.linalg.norm(proj - x)


    R_refined = Quaternion2Rotation(p[3:])
    C_refined = p[:3]
    return R_refined, C_refined