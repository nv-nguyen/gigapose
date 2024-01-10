import torch
from collections import OrderedDict

# Load the checkpoint
checkpoint_path = (
    "/home/nguyen/Documents/datasets/gigaPose_datasets/pretrained/10000.ckpt"
)
checkpoint = torch.load(checkpoint_path)
# Check if 'state_dict' exists in the checkpoint
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    state_dict = dict(state_dict)
    new_state_dict = OrderedDict()
    # Check if the attributes exist in the state_dict
    for k in state_dict:
        if k.startswith("invariant_net."):
            new_state_dict[k.replace("invariant_net", "ae_net")] = state_dict[k]

    for k in state_dict:
        if k.startswith("variant_net."):
            new_state_dict[k.replace("variant_net", "ist_net")] = state_dict[k]
    # Save the modified state_dict back to the checkpoint
    checkpoint["state_dict"] = OrderedDict(new_state_dict)

    # Save the modified checkpoint to a new file
    new_checkpoint_path = (
        "/home/nguyen/Documents/datasets/gigaPose_datasets/pretrained/10000_iter.ckpt"
    )
    torch.save(checkpoint, new_checkpoint_path)

    print(
        f"Checkpoint has been successfully modified and saved to '{new_checkpoint_path}'."
    )
else:
    print("The 'state_dict' key is not present in the checkpoint.")
