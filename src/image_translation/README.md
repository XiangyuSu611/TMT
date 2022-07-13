# Image translation
The image translation network translates the color from the exemplar to a projection of the 3D shape and the part segmentation from the projection to the exemplar.
![image](https://github.com/XiangyuSu611/TMT/blob/master/docs/image_translation_network1.png =300x)

## Get started
Pretrained VGG models download from [here](url), move them to `./models`. If you want to use our data, please download from [here](url), and place in `../data/training_data/image_translation`.
To train this network, please run this script.
```shell
python ./src/image_translation/train.py --name=rendered --dataset_mode=rendered --dataroot=./data/training_data/image_translation --niter=25 --niter_decay=35 --use_attention --maskmix --noise_for_mask --mask_epoch=35 --warp_mask_losstype=direct --PONO --PONO_C --vgg_normal_correct --batchSize=2 --gpu_ids=0 --checkpoints_dir=[checkpoint dir] --nThreads=0
```
## Acknowledgments
We thank [CocosNet](https://github.com/microsoft/CoCosNet) for their great work! 