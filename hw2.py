import os
import cv2
import numpy as np

import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature import BuildFeatureTrack
from camera_pose import EstimateCameraPose
from camera_pose import Triangulation
from camera_pose import EvaluateCheirality
from pnp import PnP_RANSAC
from pnp import PnP_nl
from reconstruction import FindMissingReconstruction
from reconstruction import Triangulation_nl
from reconstruction import RunBundleAdjustment


def visualize_3d_points(X, P_set, label):
    X = X[np.any(X != -1, axis=1)]
    m_set = None
    for P in P_set:
        T = np.eye(4)
        T[:3, :3] = P[:3, :3]
        T[:3, 3] = P[:3, 3]
        m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        m.transform(T)
        m_set = m_set + m if m_set else m
    o3d.io.write_triangle_mesh(f'camera_intermid_{label}.ply', m_set)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    pcd.colors = o3d.utility.Vector3dVector(np.full((X.shape[0], 3), 0))
    o3d.io.write_point_cloud(f'points_intermid_{label}.ply', pcd)


if __name__ == '__main__':
    K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1]
    ])
    num_images = 6
    h_im = 540
    w_im = 960

    # Load input images
    Im = np.empty((num_images, h_im, w_im, 3), dtype=np.uint8)
    for i in range(num_images):
        im_file = 'im/image{:07d}.jpg'.format(i + 1)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im[i,:,:,:] = im

    # Build feature track
    track = BuildFeatureTrack(Im, K)

    track1 = track[0,:,:]
    track2 = track[1,:,:]

    # Estimate ﬁrst two camera poses
    R, C, X = EstimateCameraPose(track1, track2)

    output_dir = 'output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set of camera poses
    P = np.zeros((num_images, 3, 4))
    # Set first two camera poses
    P[0] = np.hstack([np.eye(3), np.zeros((3, 1))])
    P[1] = R @ np.hstack([np.eye(3), -C.reshape(-1, 1)])

    ransac_n_iter = 200
    ransac_thr = 0.01
    for i in range(2, num_images):
        mask_new_point_i = FindMissingReconstruction(X, track[i])

        # Estimate new camera pose
        X_valid = X[mask_new_point_i]
        track_i_valid = track[i, mask_new_point_i]
        R, C, inlier = PnP_RANSAC(X_valid, track_i_valid, ransac_n_iter, ransac_thr)
        R_refined, C_refined = PnP_nl(R, C, X_valid[inlier], track_i_valid[inlier])

        # Add new camera pose to the set
        P[i] = R @ np.hstack([np.eye(3), -C.reshape(-1, 1)])

        for j in range(i):
            # Fine new points to reconstruct
            mask_new_point_j = FindMissingReconstruction(X, track[j])
            mask_new_point = mask_new_point_i & mask_new_point_j

            # Triangulate points
            track_new_i = track[i, mask_new_point]
            track_new_j = track[j, mask_new_point]
            X_new = Triangulation(P[i], P[j], track_new_i, track_new_j)
            X_refined = Triangulation_nl(X_new, P[i], P[j], track_new_i, track_new_j)

            # Filter out points based on cheirality
            valid_index = EvaluateCheirality(P[i], P[j], X_refined)

            # Update 3D points
            new_point_indices = np.where(mask_new_point)[0]
            valid_index_global = new_point_indices[valid_index]
            X[valid_index_global] = X_refined[valid_index]
        
        # Run bundle adjustment
        valid_ind = X[:, 0] != -1
        X_ba = X[valid_ind, :]
        track_ba = track[:i + 1, valid_ind, :]
        P_new, X_new = RunBundleAdjustment(P[:i + 1, :, :], X_ba, track_ba)
        P[:i + 1, :, :] = P_new
        X[valid_ind, :] = X_new

        P[:i+1,:,:] = P_new
        X[valid_ind,:] = X_new

        visualize_3d_points(X_valid, P[:i], i)

        ###############################################################
        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i+1):
            R_d = P[j, :, :3]
            C_d = -R_d.T @ P[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh('{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_new_h = np.hstack([X_new, np.ones((X_new.shape[0],1))])
        colors = np.zeros_like(X_new)
        for j in range(i, -1, -1):
            x = X_new_h @ P[j,:,:].T
            x = x / x[:, 2, np.newaxis]
            mask_valid = (x[:,0] >= -1) * (x[:,0] <= 1) * (x[:,1] >= -1) * (x[:,1] <= 1)
            uv = x[mask_valid,:] @ K.T
            for k in range(3):
                interp_fun = RectBivariateSpline(np.arange(h_im), np.arange(w_im), Im[j,:,:,k].astype(float)/255, kx=1, ky=1)
                colors[mask_valid, k] = interp_fun(uv[:,1], uv[:,0], grid=False)

        ind = np.sqrt(np.sum(X_ba ** 2, axis=1)) < 200
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_new[ind]))
        pcd.colors = o3d.utility.Vector3dVector(colors[ind])
        o3d.io.write_point_cloud('{}/points_{}.ply'.format(output_dir, i+1), pcd)
