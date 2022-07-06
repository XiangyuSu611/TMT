# Data generation

## Datesets
We use collections of photographs, shapes and materials for training our neural networks. We try to publish all the data needed to run our code, but due to copyright restrictions, some commercial data require users to download directly from source.
### Photographs
We use photographs collected by [PhotoShape](https://github.com/keunhong/photoshape#exemplar-images), which have been cropped and centered. These codes can be found in `/src/data_preprocess/exemplar/`. 
Please download pre-processed photos for chairs from [here](url-to-exemplar), and decompress it to `/data/exemplars/`.
### Shapes
For 3D shapes, we use [PartNet](https://partnet.cs.stanford.edu/) dataset, which contains no texture but fine-gained semantic segmentation. For each model in PartNet, we merged the individual part segmentations together and recalculated the UVs using blender. These codes can be found in `/src/data_preprocess/shape/`.
Please download pre-processed shapes for chairs from [here](url-to-shape), and decompress it to `/data/shapes/`.
### Materials
We have collected 600 photorealistic materials from differernt sources, some of which are free and some are commercially available. Unfortunately, we only have permission to publish free data, please download these materials from [here](), and the link to paid materials can be obtained from [here](). Materials should be placed in `/data/materials/`.

## Generate training data
We 


