import torch
import numpy as np
import torch.nn.init as init
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_checkpoint(model, checkpoint_path, checkpoint_key=None, prefix=""):
    checkpoint = torch.load(checkpoint_path)
    if checkpoint_key is not None:
        pretrained_dict = checkpoint[checkpoint_key]  # "state_dict"
    else:
        pretrained_dict = checkpoint
    pretrained_dict = {k.replace(prefix, ""): v for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()

    # compare keys and update value
    pretrained_dict_can_load = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    pretrained_dict_cannot_load = [
        k for k, v in pretrained_dict.items() if k not in model_dict
    ]

    model_dict_not_update = [
        k for k, v in model_dict.items() if k not in pretrained_dict
    ]

    module_cannot_load = np.unique(
        [k.split(".")[0] for k in pretrained_dict_cannot_load]  #
    )

    module_not_update = np.unique([k.split(".")[0] for k in model_dict_not_update])

    logger.info(f"Cannot load: {module_cannot_load}")
    logger.info(f"Not update: {module_not_update}")

    size_pretrained = len(pretrained_dict)
    size_pretrained_can_load = len(pretrained_dict_can_load)
    size_pretrained_cannot_load = len(pretrained_dict_cannot_load)
    size_model = len(model_dict)
    logger.info(
        f"Pretrained: {size_pretrained}/ Loaded: {size_pretrained_can_load}/ Cannot loaded: {size_pretrained_cannot_load} VS Current model: {size_model}"
    )
    model_dict.update(pretrained_dict_can_load)
    model.load_state_dict(model_dict)
    logger.info("Load pretrained done!")


def init_weights(self, init_type="xavier_uniform", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "xavier_uniform":
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    self.apply(init_func)
    # propagate to children
    for n, m in self.named_children():
        m.apply(init_weights)
