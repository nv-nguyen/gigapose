<div align="center">
<h2>
GigaPose: Fast and Robust Novel Object Pose Estimation 

via One Correspondence
<p></p>
</h2>

<h3>
<a href="https://nv-nguyen.github.io/" target="_blank"><nobr>Van Nguyen Nguyen</nobr></a> &emsp;
<a href="http://imagine.enpc.fr/~groueixt/" target="_blank"><nobr>Thibault Groueix</nobr></a> &emsp;
<a href="https://people.epfl.ch/mathieu.salzmann" target="_blank"><nobr>Mathieu Salzmann</nobr></a> &emsp;
<a href="https://vincentlepetit.github.io/" target="_blank"><nobr>Vincent Lepetit</nobr></a>

<p></p>

<a href="https://nv-nguyen.github.io/gigaPose/"><img 
src="https://img.shields.io/badge/-Webpage-blue.svg?colorA=333&logo=html5" height=28em></a>
<a href="https://arxiv.org/abs/2311.14155"><img 
src="https://img.shields.io/badge/-Paper-blue.svg?colorA=333&logo=arxiv" height=28em></a>
<a href="https://drive.google.com/file/d/11V9J4voUkovMIFxOeDCkaO7uf9EfCTZ0/view?usp=sharing"><img 
src="https://img.shields.io/badge/-SuppMat-blue.svg?colorA=333&logo=drive" height=28em></a>
<p></p>

<p align="center">
  <img src=./media/qualitative.png width="100%"/>
</p>

</h4>
</div>

**TL;DR**: GigaPose is a "hybrid" template-patch correspondence approach to estimate 6D pose of novel objects in RGB images: GigaPose first uses templates, rendered images of the CAD models, to recover the out-of-plane rotation (2DoF) and then uses patch correspondences to estimate the remaining 4DoF. 



### News 📣
- [January 19th, 2024] We released the intructions for estimating pose for custom objects and for a single reference image setting on LM-O dataset.
- [January 11th, 2024] We released the code for both training and testing settings. We are working on the demo for custom objects including detecting novel objects with [CNOS](https://github.com/nv-nguyen/cnos) and novel object pose estimation from a single reference image by reconstructing objects with [Wonder3D](https://github.com/xxlong0/Wonder3D). Stay tuned!
## Citations
``` Bash
@article{nguyen2023gigaPose,
    title={GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence},
    author={Nguyen, Van Nguyen and Groueix, Thibault and Salzmann, Mathieu and Lepetit, Vincent},
    journal={arXiv preprint arXiv:2311.14155}, 
    year={2023}}
```
GigaPose's codebase is mainly derived from [MegaPose](https://github.com/megapose6d/megapose6d):
``` Bash
@inproceedings{labbe2022megapose,
    title     = {MegaPose: 6D Pose Estimation of Novel Objects via Render \& Compare},
    author    = {Labb\'e, Yann and Manuelli, Lucas and Mousavian, Arsalan and Tyree, Stephen and Birchfield, Stan and Tremblay, Jonathan and Carpentier, Justin and Aubry, Mathieu and Fox, Dieter and Sivic, Josef},
    booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
    year      = {2022},
} 
```

## Installation :construction_worker:

<details><summary>Click to expand</summary>

### Environment
```
conda env create -f environment.yml
conda activate gigapose
bash src/scripts/install_env.sh

# to install megapose
pip install -e .

# to install bop_toolkit 
pip install git+https://github.com/thodan/bop_toolkit.git
```

### Checkpoints
```
# download testing datasets of BOP challenge
python -m src.scripts.download_test

# download cnos detections
python -m src.scripts.download_cnos

# download gigaPose's checkpoints 
python -m src.scripts.download_gigapose

# download megapose's checkpoints
python -m src.scripts.download_megapose
```

### Datasets
All datasets are defined in [BOP format](https://bop.felk.cvut.cz/datasets/). 
```
# download testing images and CAD models
python -m src.scripts.download_test

# download cnos detections
python -m src.scripts.download_cnos
```

We provide the pre-rendered templates (from [this link](https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/templates.zip)) and also the code to render the templates from the CAD models.
```
# option 1: download pre-rendered templates 
python -m src.scripts.download_bop_templates

# option 2: render templates from CAD models 
python -m src.scripts.render_bop_templates
```

Here is the structure of $ROOT_DIR after downloading all the above files:
```
├── $ROOT_DIR
    ├── datasets/ 
      ├── lmo/ 
      ├── ... 
      ├── templates/
    ├── pretrained/ 
      ├── gigaPose_v1.ckpt 
      ├── megapose-models/
```

[Optional] We also provide the training code/datasets which is not necessary for testing purposes.
<details><summary>Click to expand</summary>

```
# download training images (> 2TB)
python -m src.scripts.download_train_metaData
python -m src.scripts.download_train_cad 
python -m src.scripts.download_train 

# render templates ( 162 imgs/obj takes ~30mins for gso, ~20hrs for shapenet)
python -m src.scripts.render_gso_templates 
python -m src.scripts.render_shapenet_templates  
```

If you have training datasets pre-downloaded, you can create a symlink to the folder containing the datasets by running:
```
ln -s /path/to/datasets/gso $ROOT/datasets/gso
```

[Optional] Trick for faster converging of ISTNetwork (in-plane, scale, translation): using pretrained weights of [LoFTR](https://drive.google.com/file/d/1kW2bQejjMlmE7FGberHrubXpE_ttX2LB/view?usp=drive_link) after Kaiming initialization. Please download the weights and put them in `$ROOT_DIR/pretrained/loftr_indoor_ot.ckpt`.

</details>

</details>

## Testing on custom objects 

Working in progress...

##  Testing on [BOP datasets](https://bop.felk.cvut.cz/datasets/) :rocket:

<p align="center">
  <img src=./media/inference.png width="100%"/>
</p>

GigaPose's coarse prediction for seven core datasets of BOP challenge is available in [this link](https://drive.google.com/file/d/1QaGNIPZyR8FOOsT35V7pWJF2VlN9_M6l/view?usp=sharing). Below are the steps to reproduce the results and evaluate with BOP toolkit.

<details><summary>Click to expand</summary>

1. Running coarse prediction on a single dataset:
```
python test.py test_dataset_name=lmo run_id=$NAME_RUN
```

2. Running refinement on a single dataset:
```
python refine.py test_dataset_name=lmo run_id=$NAME_RUN
```

3. Running all steps for all 7 core datasets of BOP challenge:
```
python -m src.scripts.eval_bop
```

3. Evaluating with [BOP toolkit](https://github.com/thodan/bop_toolkit):
```
export INPUT_DIR=DIR_TO_YOUR_PREDICTION_FILE
export FILE_NAME=NAME_PREDICTION_FILE
python bop_toolkit/scripts/eval_bop19_pose.py --renderer_type=vispy --results_path $INPUT_DIR --eval_path $INPUT_DIR --result_filenames=$FILE_NAME
```

</details>

##  Pose estimation from a single image on [LM-O](https://bop.felk.cvut.cz/datasets/) :smiley_cat:

<p align="center">
  <img src=./media/wonder3d_meshes.png width="100%"/>
</p>

<details><summary>Click to expand</summary>

To relax the need of CAD models, we can reconstruct 3D models from a single image using recent works on diffusion-based 3D reconstruction such as [Wonder3D](https://github.com/xxlong0/Wonder3D), then apply the same pipeline as GigaPose to estimate object pose. Here are the steps to reproduce the results of novel object pose estimation from a single image on LM-O dataset as shown in our paper:

- Step 1: Selecting the input reference image for each object. We provide the list of reference images in [SuppMat](https://drive.google.com/file/d/11V9J4voUkovMIFxOeDCkaO7uf9EfCTZ0/view?usp=sharing). 
- Step 2: Cropping the input image (and save the [cropping matrix](https://github.com/nv-nguyen/gigapose/blob/main/src/utils/crop.py#L49) for recovering the correct scale for reconstructed 3D models).
- Step 3: Reconstructing 3D models from the reference images using [Wonder3D](https://github.com/xxlong0/Wonder3D). Note that the output 3D models are reconstructed in the coordinate frame of input image.
- Step 4: Recovering the scale of reconstructed 3D models using the cropping matrix of Step 2. 
- Step 5: Estimating the object pose using GigaPose's pipeline. 

We provide [here](https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/wonder3d_inout.zip) the inputs and outputs of Wonder3D, [here](https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/wonder3d_mesh.zip) the reconstructed 3D models in Step 1-3 and, [this script](https://github.com/nv-nguyen/gigapose/blob/main/src/scripts/recover_scale_wonder3d.py) to recover 3D models in the correct scale. [Here](https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/lmoWonder3d.zip) is the reconstructed 3D models in the correct scale and in the GT coordinate frame discussed below (note that the GT canonical frame is only for evaluation purposes with BOP Toolkit, while for real applications, we can use the object pose in the input reference image as the canonical frame). 

<details><summary>Click to expand</summary>

### Canonical frame for bop toolkit

For all evaluations, we use [bop toolkit](https://github.com/thodan/bop_toolkit.git) which requires the estimated poses defined in the same coordinate frame of GT CAD models. Therefore, there are two options:
- Option 1: Transforming the GT CAD models to the coordinate frame of the input image and adjust the GT poses accordingly.
- Option 2: Reconstructing the 3D models, then transforming it to the coordinate frame of GT CAD models by assuming the object pose in the input reference image is known.

Given that the metrics VSD, MSSD, MSPD employed in the [bop toolkit](https://github.com/thodan/bop_toolkit.git) depend on the canonical frame of the object, and for a meaningful comparison with [MegaPose](https://github.com/megapose6d/megapose6d) and GigaPose's results using GT CAD models, we opt Option 2. 
</details>
</details>

Once the reconstructed 3D models are in the correct scale and in the GT coordinate frame, we can now estimate the object pose using GigaPose's pipeline in Step 5:

```
# download the reconstructed 3D models, test images, and test_targets_bop19.json
mkdir $ROOT_DIR/datasets/lmoWonder3d
wget https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/lmoWonder3d.zip -P $ROOT_DIR/datasets/lmoWonder3d
unzip -j $ROOT_DIR/datasets/lmoWonder3d/lmoWonder3d.zip -d $ROOT_DIR/datasets/lmoWonder3d/models -x "*/._*"

# treat lmoWonder3d as a new dataset by creating a symlink 
ln -s $ROOT_DIR/datasets/lmo/test $ROOT_DIR/datasets/lmoWonder3d/test
ln -s $ROOT_DIR/datasets/lmo/test_targets_bop19.json $ROOT_DIR/datasets/lmoWonder3d/test_targets_bop19.json

# Onboarding by rendering templates from reconstructed 3D models
python -m src.scripts.render_custom_templates custom_dataset_name=lmoWonder3d

# now, it can be tested as a normal dataset as in the previous section
python test.py test_dataset_name=lmoWonder3d run_id=$NAME_RUN
python refine.py test_dataset_name=lmoWonder3d run_id=$NAME_RUN
```


##  Training
<p align="center">
  <img src=./media/training.png width="100%"/>
</p>
<details><summary>Click to expand</summary>

```
# train on GSO (ID=0), ShapeNet (ID=1), or both (ID=2)
python train.py train_dataset_id=$ID
```

</details>

## 👩‍⚖️ License
Unless otherwise specified, all code in this repository is made available under MIT license. 

## 🤝 Acknowledgments
This code is heavily borrowed from [MegaPose](https://github.com/megapose6d/megapose6d) and [CNOS](https://github.com/nv-nguyen/cnos). 

The authors thank Jonathan Tremblay, Medéric Fourmy, Yann Labbé, Michael Ramamonjisoa and Constantin Aronssohn for their help and valuable feedbacks!