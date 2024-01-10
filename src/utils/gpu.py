import os
import multiprocessing


def terminate_processes():
    for process in multiprocessing.active_children():
        process.terminate()


def assign_gpu() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        device_ids = device_ids.split(",")
    else:
        device_ids = range(int(os.environ.get("LOCAL_WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert local_rank < len(device_ids)
    cuda_id = int(device_ids[local_rank])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    if "SLURM_JOB_NODELIST" in os.environ:
        os.environ["EGL_VISIBLE_DEVICES"] = str(cuda_id)
