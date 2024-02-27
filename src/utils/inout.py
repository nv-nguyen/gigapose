import os
import errno
import shutil
import numpy as np
import os.path as osp
import json
import sys
import pandas as pd
from tqdm import tqdm
from bop_toolkit_lib import inout
from src.utils.dataset import LMO_index_to_ID, cnos_detections
from src.utils.logging import get_logger
from pathlib import Path
import copy

logger = get_logger(__name__)
MAX_VALUES = 1e6


def get_root_project():
    return Path(__file__).absolute().parent.parent.parent


def append_lib(path):
    sys.path.append(os.path.join(path, "src"))


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def write_txt(path, list_files):
    with open(path, "w") as f:
        for idx in list_files:
            f.write(idx + "\n")
        f.close()


def open_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_json(path):
    with open(path, "r") as f:
        # info = yaml.load(f, Loader=yaml.CLoader)
        info = json.load(f)
    return info


def save_json(path, info):
    # save to json without sorting keys or changing format
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


def save_npz(path, info):
    np.savez_compressed(path, **info)


def casting_format_to_save_json(data):
    # casting for every keys in dict to list so that it can be saved as json
    for key in data.keys():
        if (
            isinstance(data[key][0], np.ndarray)
            or isinstance(data[key][0], np.float32)
            or isinstance(data[key][0], np.float64)
            or isinstance(data[key][0], np.int32)
            or isinstance(data[key][0], np.int64)
        ):
            data[key] = np.array(data[key]).tolist()
    return data


def convert_dict_to_dataframe(data_dict, column_names, convert_to_list=True):
    if convert_to_list:
        data_list = list(data_dict.items())
    else:
        data_list = data_dict
    df = pd.DataFrame(data_list, columns=column_names)
    return df


def combine(list_dict):
    output = {}
    for dict_ in list_dict:
        for field in dict_.keys():
            for name_data in dict_[field].keys():
                key = field + "_" + name_data
                assert key not in output.keys()
                output[key] = dict_[field][name_data]
    return output


def group_by_image_level(data, image_key="im_id"):
    # group the detections by scene_id and im_id
    data_per_image = {}
    for det in data:
        if isinstance(det, dict):
            dets = [det]
        else:
            dets = det
        for det in dets:
            scene_id, im_id = int(det["scene_id"]), int(det[image_key])
            key = f"{scene_id:06d}_{im_id:06d}"
            if key not in data_per_image:
                data_per_image[key] = []
            data_per_image[key].append(det)
    return data_per_image


def save_bop_results(path, results, additional_name=None):
    # https://github.com/thodan/bop_toolkit/blob/37d79c4c5fb027da92bc40f36b82ea9b7b197f1d/bop_toolkit_lib/inout.py#L292
    if additional_name is not None:
        lines = [f"scene_id,im_id,obj_id,score,R,t,time,{additional_name}"]
    else:
        lines = ["scene_id,im_id,obj_id,score,R,t,time"]
    for res in results:
        if "time" in res:
            run_time = res["time"]
        else:
            run_time = -1
        lines.append(
            "{scene_id},{im_id},{obj_id},{score},{R},{t},{time}".format(
                scene_id=res["scene_id"],
                im_id=res["im_id"],
                obj_id=res["obj_id"],
                score=res["score"],
                R=" ".join(map(str, res["R"].flatten().tolist())),
                t=" ".join(map(str, res["t"].flatten().tolist())),
                time=run_time,
            )
        )
        if additional_name is not None:
            lines[-1] += ",{}".format(res[f"{additional_name}"])
    with open(path, "w") as f:
        f.write("\n".join(lines))


def load_bop_results(path, additional_name=None):
    # https://github.com/thodan/bop_toolkit/blob/37d79c4c5fb027da92bc40f36b82ea9b7b197f1d/bop_toolkit_lib/inout.py#L249
    results = []
    if additional_name is not None:
        header = f"scene_id,im_id,obj_id,score,R,t,time,{additional_name}"
        length_line = 8
    else:
        header = "scene_id,im_id,obj_id,score,R,t,time"
        length_line = 7
    with open(path, "r") as f:
        line_id = 0
        for line in f:
            line_id += 1
            if line_id == 1 and header in line:
                continue
            else:
                elems = line.split(",")
                if len(elems) != length_line:
                    raise ValueError(
                        "A line does not have {} comma-sep. elements: {}".format(
                            length_line, line
                        )
                    )

                result = {
                    "scene_id": int(elems[0]),
                    "im_id": int(elems[1]),
                    "obj_id": int(elems[2]),
                    "score": float(elems[3]),
                    "R": np.array(
                        list(map(float, elems[4].split())), np.float64
                    ).reshape((3, 3)),
                    "t": np.array(
                        list(map(float, elems[5].split())), np.float64
                    ).reshape((3, 1)),
                    "time": float(elems[6]),
                }
                if additional_name is not None:
                    result[additional_name] = float(elems[7])
                results.append(result)
    return results


def averaging_runtime_bop_results(path, has_instance_id=False):
    results = load_bop_results(path, has_instance_id)
    times = {}
    # calculate mean time for each scene_id and im_id
    for result in results:
        result_key = "{:06d}_{:06d}".format(result["scene_id"], result["im_id"])
        if result_key not in times.keys():
            times[result_key] = []
        times[result_key].append(result["time"])
    for key in times.keys():
        times[key] = np.mean(times[key])
    # replace time in results
    for result in results:
        result_key = "{:06d}_{:06d}".format(result["scene_id"], result["im_id"])
        result["time"] = times[result_key]
    # save to new file
    save_bop_results(path, results, has_instance_id)
    # logger.info(f"Averaged and saved predictions to {path}")


def calculate_runtime_per_image(results, is_refined):
    """
    Calculate the correct run_time for each image as in BOP challenge
    coarse_run_time: run_time = detection_time + total_time(all_batched_images)
    total_run_time: run_time = coarse_run_time + total_time(refinement)
    """
    # sort times by image_id
    if is_refined:
        time_names = ["time", "refinement_time"]
    else:
        time_names = ["detection_time", "time"]

    times = {}
    new_results = []
    counter = 0
    for result in results:
        result_key = "{:06d}_{:06d}".format(result["scene_id"], result["im_id"])
        if result_key not in times.keys():
            times[result_key] = {name: [] for name in time_names}
            times[result_key]["batch_id"] = []
        assert "batch_id" in result.keys(), f"batch_id is not in {result}"
        # make sure that detection_time and each batch is counted only once
        if result["batch_id"] not in times[result_key]["batch_id"]:
            times[result_key]["batch_id"].append(result["batch_id"])
            times[result_key]["time"].append(result["time"])
            if not is_refined:
                times[result_key]["detection_time"] = result["additional_time"]
            else:
                times[result_key]["refinement_time"].append(result["additional_time"])

        # delete the key additional_time and batch_id in result
        del result["additional_time"]
        del result["batch_id"]

    # calculate run_time for each image

    total_run_times = {}
    for key in times.keys():
        time = times[key]
        if not is_refined:
            total_run_time = time["detection_time"] + np.sum(time["time"])
        else:
            assert len(time["refinement_time"]) == len(time["batch_id"])
            total_run_time = np.sum(time["refinement_time"]) + np.sum(time["time"])
        total_run_times[key] = total_run_time

    # update the run_time for each image
    average_run_times = []
    for result in results:
        result_key = "{:06d}_{:06d}".format(result["scene_id"], result["im_id"])
        result["time"] = total_run_times[result_key]
        average_run_times.append(result["time"])
    logger.info(f"Average runtime per image: {np.mean(average_run_times):.3f} s")
    return results


def save_predictions_from_batched_predictions(
    prediction_dir,
    dataset_name,
    model_name,
    run_id,
    is_refined,
):
    list_files = [file for file in os.listdir(prediction_dir) if file.endswith(".npz")]
    list_files = sorted(list_files)

    name_additional_time = "detection_time" if not is_refined else "refinement_time"
    top1_predictions, topk_predictions = [], []
    instance_id = 0

    for batch_id, file in tqdm(
        enumerate(list_files), desc="Formatting predictions ..."
    ):
        data = np.load(osp.join(prediction_dir, file))
        assert len(data["poses"].shape) in [3, 4]
        is_only_top1 = len(data["poses"].shape) == 3
        if not is_only_top1:
            k = data["poses"].shape[1]

        for idx_sample in range(len(data["im_id"])):
            obj_id = int(data["object_id"][idx_sample])

            if not is_refined and "lmo" in dataset_name:
                obj_id = LMO_index_to_ID[obj_id - 1]

            if is_only_top1:
                t = data["poses"][idx_sample][:3, 3].reshape(-1)
                R = data["poses"][idx_sample][:3, :3].reshape(-1)
                score = data["scores"][idx_sample]
            else:
                t = data["poses"][idx_sample][0][:3, 3].reshape(-1)
                R = data["poses"][idx_sample][0][:3, :3].reshape(-1)
                score = data["scores"][idx_sample][0]

            top1_prediction = dict(
                scene_id=int(data["scene_id"][idx_sample]),
                im_id=int(data["im_id"][idx_sample]),
                obj_id=obj_id,
                score=score,
                t=t,
                R=R,
                time=data["time"][idx_sample],
                additional_time=data[name_additional_time][idx_sample],
                batch_id=np.copy(batch_id),
            )
            assert (
                "batch_id" in top1_prediction.keys()
            ), f"batch_id is not in {top1_prediction}"
            top1_predictions.append(top1_prediction)
            top1_prediction["instance_id"] = instance_id
            topk_predictions.append(top1_prediction.copy())

            if not is_only_top1:
                for idx_k in range(1, k):
                    t = data["poses"][idx_sample][idx_k][:3, 3].reshape(-1)
                    R = data["poses"][idx_sample][idx_k][:3, :3].reshape(-1)
                    score = data["scores"][idx_sample][idx_k]
                    topk_prediction = dict(
                        scene_id=int(data["scene_id"][idx_sample]),
                        im_id=int(data["im_id"][idx_sample]),
                        obj_id=obj_id,
                        score=score,
                        t=t,
                        R=R,
                        time=data["time"][idx_sample],
                        instance_id=instance_id,
                        additional_time=data[name_additional_time][idx_sample],
                        batch_id=np.copy(batch_id),
                    )

                    topk_predictions.append(topk_prediction)
            instance_id += 1
    name_file = f"{model_name}-pbrreal-rgb-mmodel_{dataset_name}-test_{run_id}"
    save_path = osp.join(prediction_dir, f"{name_file}.csv")
    calculate_runtime_per_image(top1_predictions, is_refined=is_refined)
    save_bop_results(
        save_path,
        top1_predictions,
        additional_name=None,
    )
    logger.info(f"Saved predictions to {save_path}")

    if not is_only_top1:
        save_path = osp.join(prediction_dir, f"{name_file}MultiHypothesis.csv")
        calculate_runtime_per_image(topk_predictions, is_refined=is_refined)
        save_bop_results(
            save_path,
            topk_predictions,
            additional_name="instance_id",
        )
        logger.info(f"Saved predictions to {save_path}")


def load_test_list_and_cnos_detections(
    root_dir, dataset_name, max_det_per_object_id=None
):
    """
    We use a sorting techniques which has been done in MegaPose (thanks Mederic Fourmy for sharing!)
    Idea: when there is no detection at object level, we use use the detections at image level
    """
    # load test list
    test_list = inout.load_json(root_dir / dataset_name / "test_targets_bop19.json")

    # load cnos detections
    cnos_dets_name = cnos_detections[dataset_name]
    cnos_dets_path = root_dir / "cnos-fastsam" / cnos_dets_name
    all_cnos_dets = inout.load_json(cnos_dets_path)

    # sort by image_id
    all_cnos_dets_per_image = group_by_image_level(all_cnos_dets, image_key="image_id")

    selected_detections = []
    for idx, test in tqdm(enumerate(test_list)):
        test_object_id = test["obj_id"]
        scene_id, im_id = test["scene_id"], test["im_id"]
        image_key = f"{scene_id:06d}_{im_id:06d}"

        # get the detections for the current image
        if image_key in all_cnos_dets_per_image:
            cnos_dets_per_image = all_cnos_dets_per_image[image_key]
            dets = [
                det
                for det in cnos_dets_per_image
                if (det["category_id"] == test_object_id)
            ]
            if len(dets) == 0:  # done in MegaPose
                dets = copy.deepcopy(cnos_dets_per_image)
                for det in dets:
                    det["category_id"] = test_object_id

            assert len(dets) > 0

            # sort the detections by score descending
            dets = sorted(
                dets,
                key=lambda x: x["score"],
                reverse=True,
            )
            # keep only the top detections
            if max_det_per_object_id is not None:
                num_instances = max_det_per_object_id
            else:
                num_instances = test["inst_count"]
            dets = dets[:num_instances]
            selected_detections.append(dets)
        else:
            logger.info(f"No detection for {image_key}")

    logger.info(f"Detections: {len(test_list)} test samples!")
    assert len(selected_detections) == len(test_list)
    selected_detections = group_by_image_level(
        selected_detections, image_key="image_id"
    )
    test_list = group_by_image_level(test_list, image_key="im_id")
    return test_list, selected_detections


def load_test_list_and_init_locs(root_dir, dataset_name, init_loc_path):
    # load test list and init locs
    test_list = inout.load_json(root_dir / dataset_name / "test_targets_bop19.json")
    try:
        init_locs = load_bop_results(init_loc_path, additional_name="instance_id")
        instance_ids = [pose["instance_id"] for pose in init_locs]
        num_instances = len(np.unique(instance_ids))
        assert len(init_locs) % num_instances == 0
        num_hypothesis = int(len(init_locs) / num_instances)
    except:
        init_locs = load_bop_results(init_loc_path)
        num_hypothesis = 1
    # sort by image_id
    all_init_locs_per_image = group_by_image_level(init_locs, image_key="im_id")
    test_list = group_by_image_level(test_list, image_key="im_id")
    return test_list, all_init_locs_per_image, num_hypothesis


if __name__ == "__main__":
    save_predictions_from_batched_predictions(
        "/home/nguyen/Documents/datasets/gigaPose_datasets/results/large_None/predictions/",
        dataset_name="icbin",
        model_name="large",
        run_id="12345678",
        is_refined=False,
    )
