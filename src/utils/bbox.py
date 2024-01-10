import numpy as np
import torch


class BoundingBox:
    def __init__(self, box, type="xyxy"):
        self.type = type
        if type == "xywh":
            self.xyxy_box = self.xywh_to_xyxy(box)
        else:
            self.xyxy_box = box
        self.box_size = None
        self.box_center = None
        self.top_left = None
        self.convert_long()

    def convert_long(self):
        if isinstance(self.xyxy_box, np.ndarray):
            self.xyxy_box = self.xyxy_box.astype(np.int32)
        else:
            self.xyxy_box = self.xyxy_box.long()

    def get_top_left(self):
        if isinstance(self.xyxy_box, np.ndarray):
            return self.xyxy_box[:2]
        elif isinstance(self.xyxy_box, torch.Tensor):
            return self.xyxy_box[:, :2]
        else:
            raise ValueError("xyxy_box must be a numpy array or torch tensor")

    def get_box_center(self):
        if self.box_center is None:
            if isinstance(self.xyxy_box, np.ndarray):
                self.box_center = np.array(
                    [
                        (self.xyxy_box[0] + self.xyxy_box[2]) / 2,
                        (self.xyxy_box[1] + self.xyxy_box[3]) / 2,
                    ]
                )

            elif isinstance(self.xyxy_box, torch.Tensor):
                self.box_center = torch.tensor(
                    [
                        (self.xyxy_box[:, 0] + self.xyxy_box[:, 2]) / 2,
                        (self.xyxy_box[:, 1] + self.xyxy_box[:, 3]) / 2,
                    ]
                )
            else:
                raise ValueError("xyxy_box must be a numpy array or torch tensor")
        return self.box_center

    def get_box_size(self):
        if self.box_size is None:
            if isinstance(self.xyxy_box, np.ndarray):
                self.box_size = np.array(
                    [
                        self.xyxy_box[2] - self.xyxy_box[0],
                        self.xyxy_box[3] - self.xyxy_box[1],
                    ]
                )
            elif isinstance(self.xyxy_box, torch.Tensor):
                assert len(self.xyxy_box.shape) == 2
                self.box_size = torch.zeros(
                    (self.xyxy_box.shape[0], 2), dtype=torch.int32
                )
                self.box_size[:, 0] = self.xyxy_box[:, 2] - self.xyxy_box[:, 0]
                self.box_size[:, 1] = self.xyxy_box[:, 3] - self.xyxy_box[:, 1]
            else:
                raise ValueError("xyxy_box must be a numpy array or torch tensor")
        return self.box_size

    def make_box_dividable(self, dividable_size, ceil=True):
        box_size = self.get_box_size()
        if isinstance(self.xyxy_box, np.ndarray):
            if ceil:
                new_size = np.ceil(box_size / dividable_size) * dividable_size
            else:
                new_size = np.floor(box_size / dividable_size) * dividable_size

            new_box = np.array(self.xyxy_box)
            new_box[2:] = new_box[:2] + new_size
        else:
            if ceil:
                new_size = (
                    torch.ceil(box_size.float() / dividable_size) * dividable_size
                )
            else:
                new_size = (
                    torch.floor(box_size.float() / dividable_size) * dividable_size
                )
            new_size = torch.clamp(new_size, min=dividable_size)
            new_box = self.xyxy_box.clone()
            new_box[:, 2:] = new_box[:, :2] + new_size
        return BoundingBox(new_box)

    def xyxy_to_xywh(self):
        if len(self.xyxy_box.shape) == 1:
            """Convert [x1, y1, x2, y2] box format to [x, y, w, h] format."""
            x1, y1, x2, y2 = self.xyxy_box
            return np.array([x1, y1, x2 - x1, y2 - y1])
        elif len(self.xyxy_box.shape) == 2:
            x1, y1, x2, y2 = (
                self.xyxy_box[:, 0],
                self.xyxy_box[:, 1],
                self.xyxy_box[:, 2],
                self.xyxy_box[:, 3],
            )
            return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        else:
            raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4)")

    def reset(self, index):
        return BoundingBox(self.xyxy_box.clone()[index])

    @staticmethod
    def xywh_to_xyxy(box):
        """Convert [x, y, w, h] box format to [x1, y1, x2, y2] format."""
        if len(box.shape) == 1:
            x, y, w, h = box
            return np.array([x, y, x + w, y + h])
        elif len(box.shape) == 2:
            x, y, w, h = (
                box[:, 0],
                box[:, 1],
                box[:, 2],
                box[:, 3],
            )
            if isinstance(box, np.ndarray):
                return np.stack([x, y, x + w, y + h], axis=1)
            elif isinstance(box, torch.Tensor):
                return torch.stack([x, y, x + w, y + h], dim=1)
            else:
                raise ValueError("box must be a numpy array or torch tensor")
        else:
            raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4)")

    def is_valid(self, image_size, min_box_size=None):
        if isinstance(self.xyxy_box, np.ndarray):
            is_inside = True
            if self.xyxy_box[0] < 0 or self.xyxy_box[1] < 0:
                is_inside = False
            if self.xyxy_box[2] >= image_size[1] or self.xyxy_box[3] >= image_size[0]:
                is_inside = False
        elif isinstance(self.xyxy_box, torch.Tensor):
            y1 = self.xyxy_box[:, 0] >= 0
            x1 = self.xyxy_box[:, 1] >= 0
            y2 = self.xyxy_box[:, 2] < image_size[1]
            x2 = self.xyxy_box[:, 3] < image_size[0]
            is_inside = y1 & x1 & y2 & x2
            is_inside = is_inside.to(self.xyxy_box.device)
            if min_box_size is not None:
                assert isinstance(self.xyxy_box, torch.Tensor)
                box_size = self.get_box_size()
                min_size = torch.min(box_size, dim=-1)[0]
                is_inside = is_inside & (min_size >= min_box_size)
        return is_inside

    def make_bbox_square(self, output_type_int=True):
        box_size = self.get_box_size()
        if isinstance(self.xyxy_box, np.ndarray):
            max_box_size = np.max(box_size)
            new_bbox = np.copy(self.xyxy_box)

            # Add padding into y axis
            displacement_x = (max_box_size - box_size[1]) / 2
            new_bbox[1] -= displacement_x
            new_bbox[3] += displacement_x
            # Add padding into x axis
            displacement_y = (max_box_size - box_size[0]) / 2
            new_bbox[0] -= displacement_y
            new_bbox[2] += displacement_y
            if output_type_int:
                new_bbox = np.round(new_bbox).astype(np.int32)
            return BoundingBox(new_bbox)

        elif isinstance(self.xyxy_box, torch.Tensor):
            box_size = box_size.float()
            max_box_size = torch.max(box_size, dim=-1)[0]
            new_bbox = self.xyxy_box.clone().float()

            # Add padding into y axis
            displacement_x = (max_box_size - box_size[:, 1]) / 2
            new_bbox[:, 1] -= displacement_x
            new_bbox[:, 3] += displacement_x
            # Add padding into x axis
            displacement_y = (max_box_size - box_size[:, 0]) / 2
            new_bbox[:, 0] -= displacement_y
            new_bbox[:, 2] += displacement_y

            new_box = BoundingBox(new_bbox)
            return new_box

    def apply_augmentation(
        self, image_size, scale_range=[0.9, 1.1], shift_range=[-0.05, 0.05]
    ):
        """
        Randomly augment a box by scaling and shifting it.
        """
        max_image_size = np.max(image_size)
        scale = np.random.uniform(scale_range[0], scale_range[1])
        shift = np.random.uniform(
            shift_range[0] * max_image_size, shift_range[1] * max_image_size, size=2
        ).astype(np.int32)

        # Scale box
        new_box = BoundingBox(self.xyxy_box.copy())
        new_box_size = np.array(new_box.get_box_size()) * scale
        new_box.xyxy_box[2:] = new_box[:2] + new_box_size

        # Shift box but keep it inside the image
        new_box.xyxy_box += [shift[1], shift[0], shift[1], shift[0]]

        # return original box if augmented box is out of image
        if not new_box.is_valid(image_size):
            return new_box
        else:
            return BoundingBox(self.xyxy_box.copy())


def compute_iou_box(input_boxes, gt_boxes):
    """Compute IoU between N boxes and K boxes.
    Args:
        input_boxes: (N, 4) [x1, y1, x2, y2]
        gt_boxes: (K, 4) [x1, y1, x2, y2]
    """

    iou = np.zeros((input_boxes.shape[0], gt_boxes.shape[0]))
    for i in range(input_boxes.shape[0]):
        for j in range(gt_boxes.shape[0]):
            # Compute intersection
            x1 = max(input_boxes[i, 0], gt_boxes[j, 0])
            y1 = max(input_boxes[i, 1], gt_boxes[j, 1])
            x2 = min(input_boxes[i, 2], gt_boxes[j, 2])
            y2 = min(input_boxes[i, 3], gt_boxes[j, 3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)

            # Compute union
            area_input = (input_boxes[i, 2] - input_boxes[i, 0]) * (
                input_boxes[i, 3] - input_boxes[i, 1]
            )
            area_gt = (gt_boxes[j, 2] - gt_boxes[j, 0]) * (
                gt_boxes[j, 3] - gt_boxes[j, 1]
            )
            union = area_input + area_gt - intersection

            # Compute IoU
            iou[i, j] = intersection / union
    return iou
