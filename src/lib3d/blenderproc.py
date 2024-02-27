import blenderproc as bproc
import numpy as np
import argparse
import os
import sys
from PIL import Image
import logging


def render_blender_proc(
    cad_path,
    output_dir,
    obj_poses,
    img_size,
    intrinsic,
    recenter_origin=False,
    is_tless=False,
):
    bproc.init()

    cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
        np.eye(4), ["X", "-Y", "-Z"]
    )
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_intrinsics_from_K_matrix(intrinsic, img_size[1], img_size[0])

    light_locations = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [0, 1]:
                light_locations.append([x, y, z])

    for location in light_locations:
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(location)
        light.set_energy(50)

    # load the objects into the scene
    obj = bproc.loader.load_obj(cad_path)[0]
    if recenter_origin:
        # recenter origin of object at center of its bounding box
        import bpy

        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    if is_tless:
        mat = obj.get_materials()[0]
        grey_col = 0.4  # np.random.uniform(0.1, 0.9)
        mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
    else:
        # Use vertex color for texturing
        for mat in obj.get_materials():
            mat.map_vertex_color()

    obj.set_cp("category_id", 1)
    # activate normal and distance rendering
    bproc.renderer.enable_distance_output(True)
    # set the amount of samples, which should be used for the color rendering
    bproc.renderer.set_max_amount_of_samples(100)
    bproc.renderer.set_output_format(enable_transparency=True)

    # activate depth rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    for idx_frame, obj_pose in enumerate(obj_poses):
        obj.set_local2world_mat(obj_pose)
        data = bproc.renderer.render()
        data.update(
            bproc.renderer.render_segmap(map_by="class", use_alpha_channel=True)
        )
        rgb = Image.fromarray(np.uint8(data["colors"][0])).convert("RGBA")
        rgb.save(os.path.join(output_dir, "{:06d}.png".format(idx_frame)))
        # bpy.ops.wm.save_as_mainfile(
        #     filepath="./blender_file.blend"
        # )
        mask = np.array(rgb.getchannel("A"))
        depth = data["depth"][0] * 1000.0  # convert to mm
        depth[mask< 255/2] = 0
        depth = np.uint16(depth)
        depth = Image.fromarray(depth)
        depth.save(os.path.join(output_dir, "{:06d}_depth.png".format(idx_frame)))
    # bproc.writer.write_bop(output_dir, depth, data["colors"], m2mm=True, append_to_existing_output=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("cad_path", nargs="?", help="Path to the model file")
    parser.add_argument("obj_pose", nargs="?", help="Path to the model file")
    parser.add_argument(
        "output_dir", nargs="?", help="Path to where the final files will be saved"
    )
    parser.add_argument("gpus_devices", nargs="?", help="GPU devices")
    parser.add_argument("disable_output", nargs="?", help="Disable output of blender")
    parser.add_argument(
        "scale_translation", nargs="?", help="scale translation to meter"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus_devices)
    os.environ["EGL_VISIBLE_DEVICES"] = str(args.gpus_devices)

    poses = np.load(args.obj_pose)
    # we can increase high energy for lightning but it's simpler to change just scale of the object to meter
    poses[:, :3, :3] = poses[:, :3, :3] / 1000.0
    poses[:, :3, 3] = poses[:, :3, 3] / 1000.0

    K = np.array([572.4114, 0.0, 320, 0.0, 573.57043, 240, 0.0, 0.0, 1.0]).reshape(
        (3, 3)
    )

    if "tless" in args.output_dir:
        is_tless = True
    else:
        is_tless = False

    if args.disable_output == "true":
        # redirect output to log file
        logfile = os.path.join(args.output_dir, "render.log")
        open(logfile, "a").close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

    # scale_meter do not change the binary mask but recenter_origin change it
    render_blender_proc(
        args.cad_path,
        args.output_dir,
        poses,
        intrinsic=K,
        img_size=[480, 640],
        recenter_origin=True,
        is_tless=is_tless,
    )
    if args.disable_output == "true":
        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)
        os.system("rm {}".format(logfile))
