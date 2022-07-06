# Data generation

## Datesets
We use collections of photographs, shapes and materials for training our neural networks. We try to publish all the data needed to run our code, but due to copyright restrictions, some commercial data require users to download directly from source.
### Photographs
We use photographs collected by [PhotoShape](https://github.com/keunhong/photoshape#exemplar-images), which have been cropped and centered. These codes can be found in `./preprocess/exemplars/`. 
Please download pre-processed photos for chairs from [here](url-to-exemplar), and decompress it to `/data/exemplars/`.
### Shapes
For 3D shapes, we use [PartNet](https://partnet.cs.stanford.edu/) dataset, which contains no texture but fine-gained semantic segmentation. For each model in PartNet, we merged the individual part segmentations together and recalculated the UVs using blender. These codes can be found in `./preprocess/shapes/`.
Please download pre-processed shapes for chairs from [here](url-to-shape), and decompress it to `/data/shapes/`.
### Materials
We have collected 600 photorealistic materials from differernt sources, some of which are free and some are commercially available. Unfortunately, we only have permission to publish free data, please download these materials from [here](), and the link to paid materials can be obtained from [here](). Materials should be placed in `/data/materials/`.

## Generate training data
We imporved PhotoShape's data generation pipeline to generate our training data.
1️⃣ We align exemplar photos and shapes to get numbers of image-shape pairs, which is related to `./generation/pairs/`.
2️⃣ Based on image-shape pairs, we use Photoshape's alignment methods to get an initial substance for corresponding 3D shape's parts, and group different semantic parts to obtain more realistic results. This is related to `./generation/substances/`.
3️⃣ After that, we randomly sample a material of the corresponding category for each part, generate blender files and render blender files from different views to get (rendering, segmentation) pairs, which will be used to train our network.


