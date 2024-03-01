import os
import os.path as osp
import hydra
from omegaconf import DictConfig, OmegaConf
from bop_toolkit_lib import inout
from functools import partial
import multiprocessing
from tqdm import tqdm
from src.utils.time import Timer
from src.utils.logging import get_logger, start_disable_output, stop_disable_output

logger = get_logger(__name__)
timer = Timer()


def eval_bop(idx, dataset_name, list_prefix, list_predictions, results_dir):
    pred_path = list_predictions[idx]
    input_dir, file_name = os.path.dirname(pred_path), os.path.basename(pred_path)

    if "hb" in input_dir or "itodd" in input_dir:
        # copy file to results since no gt available
        new_file_name = file_name.replace(".csv", f"_{list_prefix[idx]}.csv")
        command = f"cp {list_predictions[idx]} {results_dir}/{new_file_name}"
        logger.info(f"Running {command}")
        os.system(command)
        return 0

    else:
        command = f"python bop_toolkit/scripts/eval_bop19_pose.py --renderer_type=vispy --results_path {input_dir} --eval_path {input_dir} --result_filenames={file_name}"
        logger.info(f"Running {command}")
        os.system(command)

        result_path = osp.join(input_dir, file_name.split(".")[0], "scores_bop19.json")
        scores = inout.load_json(result_path)
        new_result_path = f"{results_dir}/{dataset_name}_{list_prefix[idx]}.json"
        command = f"cp {result_path} {new_result_path}"
        logger.info(f"Running {command}")
        os.system(command)
        return {list_prefix[idx]: scores}


def run_bop(cfg: DictConfig) -> None:
    timer.tic()
    logger.info(f"Evaluating {cfg.dataset_name}")
    prefix = f"{cfg.model.model_name}_{cfg.dataset_name}"
    log = start_disable_output(f"{cfg.results_dir}/log.txt")

    # run coarse estimation
    coarse_command = f"python test.py disable_output=False machine={cfg.machine.name} test_dataset_name={cfg.dataset_name} run_id={cfg.dataset_name}{cfg.run_id}"
    if cfg.machine.dryrun is False:
        coarse_command += " machine.dryrun=False"
    logger.info(f"Running {coarse_command}")
    os.system(coarse_command)

    # run refiner with top 1
    refine_command = f"python refine.py machine={cfg.machine.name} test_dataset_name={cfg.dataset_name} name_exp={prefix}{cfg.run_id} use_multiple=False"
    logger.info(f"Running {refine_command}")
    os.system(refine_command)

    # run refine with top 5
    refine_multiple_command = refine_command.replace(
        "use_multiple=False", "use_multiple=True"
    )
    logger.info(f"Running {refine_multiple_command}")
    os.system(refine_multiple_command)

    # bop evaluation in parallel
    save_dir = f"{cfg.save_dir}/{cfg.model.model_name}_{cfg.dataset_name}{cfg.run_id}"
    folders = ["predictions", "refined_predictions", "refined_multiple_predictions"]
    list_predictions = []
    for folder in folders:
        files = [
            osp.join(save_dir, folder, f)
            for f in os.listdir(f"{save_dir}/{folder}")
            if f.endswith(".csv")
        ]
        list_predictions.append(sorted(files)[0])
    logger.info(f"Found {len(list_predictions)} predictions")

    # input_dir, file_name = os.path.dirname(file), os.path.basename(file)
    pool = multiprocessing.Pool(processes=3)
    logger.info("Start running evaluation with BOP toolkit!")
    eval_bop_with_index = partial(
        eval_bop,
        dataset_name=cfg.dataset_name,
        list_prefix=["coarse", "refined", "refined_multiple"],
        list_predictions=list_predictions,
        results_dir=cfg.results_dir,
    )
    values = list(
        tqdm(
            pool.imap_unordered(eval_bop_with_index, range(len(list_predictions))),
            total=len(list_predictions),
        )
    )
    stop_disable_output(log)
    if cfg.dataset_name not in ["hb", "itodd"]:
        scores = {}
        for value in values:
            for name in value:
                scores[name] = value[name]["bop19_average_recall"]
        logger.info(f"Results for {cfg.dataset_name}: {scores}")
    logger.info(f"Eval done in {timer.toc()} s")
    logger.info("---" * 100)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="test",
)
def run_seven_cores(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    name_eval = "GigaPose"
    cfg.save_dir = osp.join(cfg.machine.root_dir, "results")
    cfg.results_dir = f"results/{name_eval}"
    cfg.run_id = name_eval
    os.makedirs(cfg.results_dir, exist_ok=True)

    for dataset_name in [
        "lmo",
        "tudl",
        "icbin",
        "tless",
        "ycbv",
        "itodd",  # gt not available
        "hb", # gt not available
    ]:
        logger.info(f"Eval {dataset_name}")
        cfg.dataset_name = dataset_name
        run_bop(cfg)
        logger.info("---" * 100)


if __name__ == "__main__":
    run_seven_cores()
