import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms._functional_tensor as F_t
from PIL import Image

img = Image.open("./data/lm.png")
img_pil_rot = img.rotate(-45)
img_pil_rot.save("./data/lm_rot.png")

img_tensor = T.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0).repeat(4, 1, 1, 1)
matrix = F._get_inverse_affine_matrix(
    center=[0.0, 0.0], angle=45, translate=(100, 100), scale=1, shear=[0.0, 0.0]
)
img_tensor_rot = F.affine(img_tensor, angle=45, translate=(0, 0), scale=1, shear=0)
img_tensor_rot = F_t.affine(
    img_tensor,
    matrix=matrix,
)
for i in range(1):
    img_tensor_rot_pil = T.ToPILImage()(img_tensor_rot[i])
    img_tensor_rot_pil.save(f"./data/lm_rot_torch{i}.png")
# affine = T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
# print(affine)
