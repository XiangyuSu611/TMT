# Data generation

## Datesets
We use collections of **photographs**, **shapes** and **materials** for training our neural networks. We try to publish all the data needed to run our code, but due to copyright restrictions, some commercial data require users to download directly from source.
### Photographs
We use photographs collected by [PhotoShape](https://github.com/keunhong/photoshape#exemplar-images), which have been cropped and centered. These codes can be found in `./preprocess/exemplars/`. It should be noticed that we use [MINC](http://opensurfaces.cs.cornell.edu/publications/minc/) to get pixel-wised substance prediction, pre-trained model can be downloaded from [here]().

If you want to use our pre-processed photos for chairs, please download from [here](url-to-exemplar), and decompress it to `../data/exemplars/`.

Structure of exemplar files should look like this:
```Python
exemplars/1
├── original.jpg   
├── cropped.jpg
├── image
│   ├── substance_map_minc_vgg16.map.v2.png # predicted pixel-wised substance map.
│   └── substance_map_minc_vgg16.vis.v2.png # visualization of predicted pixel-wised substance map.
└── numpy
    └──align_hog_8.npz  # pre-computed HOG feature.
```
### Shapes
For 3D shapes, we use [PartNet](https://partnet.cs.stanford.edu/), which contains no texture but fine-gained semantic segmentation. For each model in PartNet, we merged the individual part segmentations together and recalculated the UVs using blender. These codes can be found in `./preprocess/shapes/`.

Please download pre-processed shapes for chairs from [here](url-to-shape), and decompress it to `../data/shapes/`.

Structure of shape files should look like this:
```Python
shapes/1
├── images/aligment/rendering
│   ├── fov=50,theta=0.00000000,phi=0.78539816.png  # rendering from different point of views.
│   └── *** 
├── models
│   ├── uvmapped_v2.obj
│   └── uvmapped_v2.mtl
└── numpy
    └── align_hog_8.npz # pre-computed HOG feature.
```
### Materials
We have collected 600 photorealistic materials from differernt sources, some of which are free and some are commercially available. Unfortunately, we only have permission to publish free data, please download these materials from [here](), and they should be placed in `../data/materials/`. In addition, `materials.json`  includes the source and name of the paid material, which can be downloaded by yourself.

If you want to collect your own material dataset, here are some useful photorealistic material dataset:
+ [Textures.com](https://www.textures.com/library)
+ [3D textures](https://3dtextures.me/)
+ [Free PBR](https://freepbr.com/)
+ [Poly Haven](https://polyhaven.com/)
+ [Poliigon](https://www.poliigon.com/)
+ [AmbientCG](https://ambientcg.com/)
+ [Sharetextures](https://www.sharetextures.com/)
+ [Vray](https://www.vray-materials.de/)

## Generate training data
We imporved PhotoShape's data generation pipeline to generate our training data.

1️⃣ We align exemplar photos and shapes to get numbers of image-shape pairs, which is related to `./preprocess/pairs/`.

Structure of pair files should look like this:
```Python
pairs/1
├── images
│   ├── shape_rend_phong_500x500.png  # rendering of shape.
│   ├── shape_fg_bbox_500x500.png  # foreground of renderings.
│   ├── shape_rend_segments_500x500.map.png  # semantic segmentiation of shape.
│   ├── shape_rend_segments_500x500.vis.png  # visualization of semantic segmentation.
│   ├── shape_segment_map_raw_500x500.png  # visualization of semantic segmentation.
│   ├── exemplar_rend_flow_silhouette_500x500.png  # visualization of flow.
│   ├── shape_warped_segments_500x500.map.v2.png  # SIFT-warped semantic segmentation.
│   ├── shape_warped_segments_500x500.vis.v2.png  # visualization of warped semantic segmentation.
│   ├── shape_clean_segments_500x500.map.v2.png  # CRF-cleaned semantic segmentation.
│   ├── shape_clean_segments_500x500.vis.v2.png  # visualization of cleaned semantic segmentation.
│   ├── pair_proxy_substances.map.png  # proxy substance segmentation of shape.
│   ├── pair_proxy_substances.vis.png  # visualization of substance segmentation.
│   ├── pair_shape_substances.map.png  # final substance segmentation of shape.
│   └── pair_shape_substances.vis.png  # visualization of final substance segmentation.
└── numpy
    └── exemplar_rend_flow_silhouette_500x500.npz  # numpy file computed by SIFT-Flow.
```

2️⃣ Based on image-shape pairs, we use Photoshape's alignment methods to get an initial substance for corresponding 3D shape's parts, which is related to `./preprocess/pairs/`. Then, we group different semantic parts to obtain more realistic results.

3️⃣ After that, we randomly sample a material of the corresponding category for each part, generate blender files and render blender files from different views to get (rendering, segmentation) image pairs, which will be used to train our network.
This is related to `./generation/renderings/`.

