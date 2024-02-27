import numpy as np
from scipy.spatial.transform import Rotation
import torch


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True))


def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0.5, 0.0, 0.0])
    # original  tmp = np.array([0.0, 0.0, -1.0])
    # print warning when camera location is parallel to tmp
    norm = min(
        np.linalg.norm(cam_location - tmp, axis=-1),
        np.linalg.norm(cam_location + tmp, axis=-1),
    )
    if norm < 1e-3:
        print("Warning: camera location is parallel to tmp")
        tmp = np.array([0.0, -1.0, 0.0])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def opencv2opengl(cam_matrix_world):
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if len(cam_matrix_world.shape) == 2:
        return np.matmul(transform, cam_matrix_world)
    else:
        transform = np.tile(transform, (cam_matrix_world.shape[0], 1, 1))
        return np.matmul(transform, cam_matrix_world)


def R_opencv2R_opengl(R_opencv):
    transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    if len(R_opencv.shape) == 2:
        return np.matmul(transform, R_opencv)
    else:
        transform = np.tile(transform, (R_opencv.shape[0], 1, 1))
        return np.matmul(transform, R_opencv)


def matrix4x4(R, T, scale_translation=1.0):
    matrix4x4 = np.eye(4)
    matrix4x4[:3, :3] = np.array(R).reshape(3, 3)
    matrix4x4[:3, 3] = np.array(T).reshape(-1) * scale_translation
    return matrix4x4


def geodesic(R1, R2):
    theta = (np.trace(R2.dot(R1.T)) - 1) / 2
    theta = np.clip(theta, -1, 1)
    return np.degrees(np.arccos(theta))


def perspective(K, obj_pose, pts):
    results = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        R, T = obj_pose[:3, :3], obj_pose[:3, 3]
        rep = np.matmul(K, np.matmul(R, pts[i].reshape(3, 1)) + T.reshape(3, 1))
        results[i, 0] = np.int32(rep[0] / rep[2])  # as matplot flip  x axis
        results[i, 1] = np.int32(rep[1] / rep[2])
    return results


def apply_transfrom(transform4x4, matrix4x4):
    # apply transform to a 4x4 matrix
    new_matrix4x4 = transform4x4.dot(matrix4x4)
    return new_matrix4x4


def rotation_from_axis_and_angle(axis, angle):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
    return torch.from_numpy(transform).float()


def spherical_to_cartesian(azimuth, elevation, radius):
    x = radius * np.sin(elevation) * np.cos(azimuth)
    y = radius * np.sin(elevation) * np.sin(azimuth)
    z = radius * np.cos(elevation)
    return np.stack((x, y, z), axis=-1)


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def rotation2quaternion(rot):
    import pyrr
    m = pyrr.Matrix33(
        [
            [rot[0], rot[1], rot[2]],
            [rot[3], rot[4], rot[5]],
            [rot[6], rot[7], rot[8]],
        ]
    )
    return m.quaternion
