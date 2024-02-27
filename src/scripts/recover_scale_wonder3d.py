import numpy as np
import os
from scipy.spatial.transform import Rotation
import trimesh
from pathlib import Path
from tqdm import tqdm


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True))


def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([1.0, 0.0, 0.0])
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


def load_fixed_poses(wonder3d_cam_pose_dir):
    view_types = ["front", "front_right", "right", "back", "left", "front_left"]
    poses = {}
    for face in view_types:
        identity = np.eye(4)
        path = wonder3d_cam_pose_dir / "%03d_%s_RT.txt" % (0, face)
        RT = np.loadtxt(path)
        identity[:3, :4] = RT
        poses[face] = opencv2opengl(identity)
    return poses


# default camera pose in wonder3d
R_inplane = Rotation.from_euler("z", -90, degrees=True).as_matrix()
matrix_inplane = np.eye(4)
matrix_inplane[:3, :3] = R_inplane
camera_pose = look_at([0, -1, 0], np.array([0, 0, 0]))
camera_pose = camera_pose.copy() @ np.linalg.inv(matrix_inplane)

# default object pose in wonder3d
object_pose_default = np.linalg.inv(camera_pose)

# default camera intrinsics in wonder3d: orthographic projection
W, H = 224, 224
K_wonder = np.array(
    [
        [W, 0, W / 2.0],
        [0, H, H / 2.0],
        [0, 0, 1],
    ]
)

if __name__ == "__main__":
    root_dir = Path("YOUR_ROOT_DIR")
    inout_dir = root_dir / "wonder3d_inout"
    mesh_dir = root_dir / "wonder3d_mesh"
    save_dir = root_dir / "models"
    save_dir.mkdir(exist_ok=True)

    for obj_id in tqdm([1, 5, 6, 8, 9, 10, 11, 12]):
        # set object pose by default
        object_pose = object_pose_default.copy()

        # read gt input data (object_pose: identity if not available, intrinsic: K, cropping: query_M)
        obj_dir = f"{inout_dir}/obj_{obj_id:06d}"
        gt_data_path = [
            os.path.join(str(obj_dir), path)
            for path in os.listdir(obj_dir)
            if path.endswith(".npz")
        ][0]
        gt_data = np.load(gt_data_path)

        # change canonical pose if object pose is available (it is identity if not available)
        object_pose[:3, :3] = np.dot(
            gt_data["object_pose"][:3, :3].T, object_pose[:3, :3]
        )

        # calculating scale_factor: see eq(5) in SuppMat
        scale_translation_reconst_to_gt = np.linalg.norm(
            gt_data["object_pose"][:3, 3] / 1000.0
        ) / np.linalg.norm(object_pose[:3, 3])

        scale_resize = 1 / gt_data["query_M"][0, 0]
        scale_focal = K_wonder[0, 0] / gt_data["K"][0, 0]
        total_scale = scale_resize * scale_focal * scale_translation_reconst_to_gt

        # set scale
        object_pose[:3, :3] *= total_scale

        # set translation
        object_pose[:3, 3] = gt_data["object_pose"][:3, 3]

        # from mm to m
        object_pose[:3, :3] *= 1000
        object_pose[:3, 3] *= 0

        # export meshes
        mesh_path = mesh_dir / f"obj_{obj_id:06d}.obj"
        new_mesh = trimesh.load(mesh_path)
        new_mesh.apply_transform(object_pose)
        new_mesh.export(save_dir / f"obj_{obj_id:06d}.obj")
