from src.utils.batch import BatchedData
from src.utils.logging import get_logger
import torch

logger = get_logger(__name__)

from torch.utils.data import DataLoader


class NoneFilteringDataLoader(DataLoader):
    def __iter__(self):
        # Get the original iterator from the parent DataLoader
        original_iterator = super(NoneFilteringDataLoader, self).__iter__()

        # Iterate over batches and filter out None batches
        for batch in original_iterator:
            if batch is not None:
                yield batch


def concat_dataloader(list_dataloaders, mode="max_size_cycle", names=None):
    logger.info(f"Concatenating dataloaders {type(list_dataloaders)}")
    from pytorch_lightning.utilities import CombinedLoader

    if isinstance(list_dataloaders, dict):
        combined_loader = CombinedLoader(list_dataloaders, mode)
        return combined_loader
    else:
        if names is None:
            names = [f"{i}" for i in range(len(list_dataloaders))]
        list_dataloaders = {
            name: loader for name, loader in zip(names, list_dataloaders)
        }
        combined_loader = CombinedLoader(list_dataloaders, mode=mode)
        return combined_loader


def collate_fn_with_batch_size(batch, batch_size):
    """Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "query", "ref", "query_keypoints" and "ref_keypoints"
    Returns:
        dict: dictionary with same keys but as batched tensors
    """
    batch_dict = {}
    missing = 0
    for sample in batch:
        if sample is None:
            missing += 1
            continue
        else:
            non_empty_sample = sample
        for key in sample.keys():
            if key not in batch_dict:
                batch_dict[key] = BatchedData(batch_size=None, data=sample[key])
            else:
                batch_dict[key].cat(sample[key])
    if (
        missing > 0
    ):  # for distributed training: handle missing and make sure it has perfect batch size
        logger.info(f"Found {missing} missing samples")
        for _ in range(missing):
            for key in non_empty_sample.keys():
                if "keypoints" in key:
                    tmp = torch.ones_like(non_empty_sample[key]) * -1
                    batch_dict[key].cat(tmp)
                else:
                    batch_dict[key].cat(non_empty_sample[key])
    for key in batch_dict:
        batch_dict[key] = batch_dict[key].data  # [:max_batch_size]
        # if batch_size is not None:
        #     assert batch_dict[key].shape[0] == batch_size, "batch size is not correct"
    assert batch_dict != {}, f"batch={batch} is empty"
    return batch_dict
