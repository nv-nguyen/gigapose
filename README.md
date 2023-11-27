<div align="center">
<h3>
GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence
<p></p>
</h3>

<h4>
<a href="https://nv-nguyen.github.io/" target="_blank"><nobr>Van Nguyen Nguyen</nobr></a> &emsp;
<a href="http://imagine.enpc.fr/~groueixt/" target="_blank"><nobr>Thibault Groueix</nobr></a> &emsp;
<a href="https://people.epfl.ch/mathieu.salzmann" target="_blank"><nobr>Mathieu Salzmann</nobr></a> &emsp;
<a href="https://vincentlepetit.github.io/" target="_blank"><nobr>Vincent Lepetit</nobr></a>

<p></p>

<a href="https://nv-nguyen.github.io/gigaPose/"><img 
src="https://img.shields.io/badge/-Webpage-blue.svg?colorA=333&logo=html5" height=20em></a>
<a href="https://arxiv.org/abs/2311.14155"><img 
src="https://img.shields.io/badge/-Paper-blue.svg?colorA=333&logo=arxiv" height=20em></a>
<p></p>

<p align="center">
  <img src=./media/overview_inference.png width="100%"/>
</p>

</h4>
</div>

**TL;DR**: GigaPose is a "hybrid" template-patch correspondence approach to estimate 6D (coarse) pose of novel objects in RGB images from a single 2D-to-2D correspondence. We experimentally show that GigaPose is (i) faster (38x for coarse pose stage), (ii) robust to segmentation errors made by CNOS, and (iii) more accurate with 3.2 AP improvement (after refinement) on seven core dataset of the BOP challenge.


If our project is helpful for your research, please consider citing : 
``` Bash
@article{nguyen2023gigaPose,
    title={GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence},
    author={Nguyen, Van Nguyen and Groueix, Thibault and Salzmann, Mathieu and Lepetit, Vincent},
    journal={arXiv preprint arXiv:2311.14155}, 
    year={2023}}
```
GigaPose's codebase is mainly derived from [MegaPose](https://github.com/megapose6d/megapose6d) and [CNOS](https://github.com/nv-nguyen/cnos). Please consider citing:
``` Bash
@inproceedings{labbe2022megapose,
    title     = {MegaPose: 6D Pose Estimation of Novel Objects via Render \& Compare},
    author    = {Labb\'e, Yann and Manuelli, Lucas and Mousavian, Arsalan and Tyree, Stephen and Birchfield, Stan and Tremblay, Jonathan and Carpentier, Justin and Aubry, Mathieu and Fox, Dieter and Sivic, Josef},
    booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
    year      = {2022},
} 

@inproceedings{nguyen2023cnos,
    title={CNOS: A Strong Baseline for CAD-based Novel Object Segmentation},
    author={Nguyen, Van Nguyen and Groueix, Thibault and Ponimatkin, Georgy and Lepetit, Vincent and Hodan, Tomas},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={2134--2140},
    year={2023}
}
```

We are working on releasing the code. Stay tuned!