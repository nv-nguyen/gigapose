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
src="https://img.shields.io/badge/-Webpage-blue.svg?colorA=333&logo=html5" height=30em></a>
<a href="https://arxiv.org/abs/2311.14155"><img 
src="https://img.shields.io/badge/-Paper-blue.svg?colorA=333&logo=arxiv" height=30em></a>
<p></p>

<p align="center">
  <img src=./media/overview_inference.png width="100%"/>
</p>

</h4>
</div>

**TL;DR**: GigaPose is a "hybrid" template-patch correspondence approach to estimate 6D pose of novel objects in RGB images: GigaPose first uses templates, rendered images of the CAD models, to recover the out-of-plane rotation (2DoF) and then uses patch correspondences to estimate the remaining 4DoF. We experimentally show that GigaPose is (i) 38x faster for coarse pose stage, (ii) robust to segmentation errors made by the 2D detector, and (iii) more accurate with 3.2 AP improvement on seven core dataset of the BOP challenge.


We are working on releasing the code. Stay tuned!