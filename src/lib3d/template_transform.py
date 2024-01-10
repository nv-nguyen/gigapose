import os
import numpy as np
import pathlib
from src.utils.inout import get_root_project
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from src.lib3d.farthest_sampling import FPS
from src.lib3d.numpy import opencv2opengl, geodesic


def extract_inplane_from_pose(pose):
    inp = Rotation.from_matrix(pose).as_euler("zyx", degrees=True)[0]
    return inp


def convert_inplane_to_rotation(inplane):
    R_inp = Rotation.from_euler("z", -inplane, degrees=True).as_matrix()
    return R_inp


def adding_inplane_to_pose(pose, inplane):
    R_inp = convert_inplane_to_rotation(inplane)
    pose = np.dot(R_inp, pose)
    return pose


def compute_inplane(rot_query_openCV, rot_template_openCV, show_warning=True):
    delta = rot_template_openCV.dot(rot_query_openCV.T)
    inp = extract_inplane_from_pose(delta)
    # double check to make sure that reconved rotation is correct
    R_inp = convert_inplane_to_rotation(inp)
    recovered_R1 = R_inp.dot(rot_template_openCV)
    err = geodesic(recovered_R1, rot_query_openCV)
    if err >= 15 and show_warning:
        print("WARINING, error of recovered pose is >=15, err=", err)
    return inp


def get_obj_poses_from_template_level(
    level, pose_distribution, return_cam=False, return_index=False
):
    root_project = get_root_project()
    if return_cam:
        obj_poses_path = os.path.join(
            root_project, f"src/lib3d/predefined_poses/cam_poses_level{level}.npy"
        )
        obj_poses = np.load(obj_poses_path)
    else:
        obj_poses_path = os.path.join(
            root_project, f"src/lib3d/predefined_poses/obj_poses_level{level}.npy"
        )
        obj_poses = np.load(obj_poses_path)

    if pose_distribution == "all":
        if return_index:
            index = np.arange(len(obj_poses))
            return index, obj_poses
        else:
            return obj_poses
    elif pose_distribution == "upper":
        cam_poses_path = os.path.join(
            root_project, f"src/lib3d/predefined_poses/cam_poses_level{level}.npy"
        )
        cam_poses = np.load(cam_poses_path)
        if return_index:
            index = np.arange(len(obj_poses))[cam_poses[:, 2, 3] >= 0]
            return index, obj_poses[cam_poses[:, 2, 3] >= 0]
        else:
            return obj_poses[cam_poses[:, 2, 3] >= 0]


def load_index_level_in_level2(level, pose_distribution):
    # created from https://github.com/nv-nguyen/DiffusionPose/blob/52e2c55b065c9637dcd284cc77a0bfb3356d218a/src/poses/find_neighbors.py
    root_repo = get_root_project()
    index_path = os.path.join(
        root_repo,
        f"src/lib3d/predefined_poses/idx_{pose_distribution}_level{level}_in_level2.npy",
    )
    return np.load(index_path)


def load_mapping_id_templates_to_idx_pose_distribution(level, pose_distribution):
    """
    Return the mapping from the id of the template to the index of the pose distribution
    """
    index_range, _ = get_obj_poses_from_template_level(
        level=level,
        pose_distribution=pose_distribution,
        return_index=True,
    )
    mapping = {}
    for i in range(len(index_range)):
        mapping[int(index_range[i])] = i
    return mapping


def load_template_poses(is_opengl_camera, dense=False):
    current_dir = pathlib.Path(__file__).parent.absolute()
    path = f"{current_dir}/predefined_poses/sphere_level"
    if dense:
        path += "3.npy"
    else:
        path += "2.npy"
    template_poses = np.load(path)
    if is_opengl_camera:
        for id_frame in range(len(template_poses)):
            template_poses[id_frame] = opencv2opengl(template_poses[id_frame])
    return template_poses


class NearestTemplateFinder(object):
    def __init__(
        self,
        level_templates,
        pose_distribution,
        return_inplane,
        normalize_query_translation=True,
    ):
        self.level_templates = level_templates
        self.normalize_query_translation = normalize_query_translation
        self.pose_distribution = pose_distribution
        self.return_inplane = return_inplane

        self.avail_index, self.obj_template_poses = get_obj_poses_from_template_level(
            level_templates, pose_distribution, return_cam=False, return_index=True
        )

        # we use the location to find for nearest template on the sphere
        self.obj_template_openGL_poses = opencv2opengl(self.obj_template_poses)

    def search_nearest_template(self, obj_query_pose):
        # convert query pose to OpenGL coordinate
        obj_query_openGL_pose = opencv2opengl(obj_query_pose)
        obj_query_openGL_location = obj_query_openGL_pose[:, 2, :3]  # Mx3
        obj_template_openGL_locations = self.obj_template_openGL_poses[:, 2, :3]  # Nx3

        # find the nearest template
        distances = cdist(obj_query_openGL_location, obj_template_openGL_locations)
        best_index_in_pose_distribution = np.argmin(distances, axis=-1)  # M
        if self.return_inplane:
            nearest_poses = self.obj_template_poses[best_index_in_pose_distribution]
            inplanes = np.zeros(len(obj_query_pose))
            for idx in range(len(obj_query_pose)):
                rot_query_openCV = obj_query_pose[idx, :3, :3]
                rot_template_openCV = nearest_poses[idx, :3, :3]
                inplanes[idx] = compute_inplane(rot_query_openCV, rot_template_openCV)
            return self.avail_index[best_index_in_pose_distribution], inplanes
        else:
            return self.avail_index[best_index_in_pose_distribution]

    def search_nearest_query(self, obj_query_pose):
        """
        Search nearest query closest to our template_pose
        """
        obj_query_openGL_pose = opencv2opengl(obj_query_pose)
        obj_query_openGL_location = obj_query_openGL_pose[:, 2, :3]  # Mx3
        obj_template_openGL_locations = self.obj_template_openGL_poses[:, 2, :3]  # Nx3
        distances = cdist(obj_template_openGL_locations, obj_query_openGL_location)
        best_index = np.argmin(distances, axis=-1)  # M
        return best_index


def farthest_sampling(openCV_poses, num_points):
    # convert query pose to OpenGL coordinate
    openGL_pose = opencv2opengl(openCV_poses)
    openGL_pose_location = openGL_pose[:, 2, :3]  # Mx3
    # apply farthest point sampling
    _, farthest_idx = FPS(openGL_pose_location, num_points).fit()
    return farthest_idx
