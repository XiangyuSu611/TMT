# Image translation
The image translation network translates the color from the exemplar to a projection of the 3D shape and the part segmentation from the projection to the exemplar.

<div align=center><img src="https://github.com/XiangyuSu611/TMT/blob/master/docs/image_translation_network.png" width="60%"></div>

## Get started
Please download pre-trained VGG models form [here](https://drive.google.com/file/d/1ENUkS_zKy3dbb3j9yA5k3WoHzYS4FNck/view?usp=sharing), move them to `./models`.
To train this network, please run this script.
```shell
python ./src/image_translation/train.py --name=rendered --dataset_mode=rendered --dataroot=./data/training_data/image_translation --niter=25 --niter_decay=25 --use_attention --maskmix --noise_for_mask --mask_epoch=25 --warp_mask_losstype=direct --PONO --PONO_C --vgg_normal_correct --batchSize=2 --gpu_ids=0 --checkpoints_dir=[checkpoint dir] --nThreads=0
```
## Acknowledgments
We thank [CocosNet](https://github.com/microsoft/CoCosNet) for their great work!