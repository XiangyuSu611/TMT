# Material transfer

Use pre-trained image translation and material prediction network, we can get final material transfer results in this part.

## Pre-process

Our pipeline starts with a photo and a 3D shape with semantic segmentation.

For wild photo, we should remove background and estimate camera pose. There are many ways to remove the background, if you don't want to configure addtional environment, we recommend [remove.bg](https://www.remove.bg/). And we recommend to move object to the center of image for better results.

````shell
python ./material_transfer/pre_process/crop_and_center.py
````

To estimate the camera pose, we use pre-trained camera pose estimator from [DISN](https://github.com/laughtervv/DISN),  please download this [model](https://drive.google.com/file/d/11uo5-_QKQPeGGidzLPhcvEndp2c19Nzi/view?usp=sharing), place in `./cam_estimation/checkpoint/`, and run this script.

````shell
python ./material_transfer/cam_estimation/est_cam_syn.py --cam_est --cam_log_dir /home/code/TMT/src/material_transfer/cam_estimation/checkpoint/
````

Next, we projection 3D shape from estimated camera pose to get semantic projection, and generate test pairs, these pairs will be saved at `./exemplar/validation`.

``````shell
python ./material_transfer/pre_process/generate_test_pair.py 
``````

## Run network
Please download our pre-trained image translation network from [here](https://drive.google.com/file/d/1g60rZ8K-Oz9o2hswI4wFtZAT79J2U-cW/view?usp=sharing), and place them in `../image_translation/checkpoint/`; for material prediction network, please download from [here](https://drive.google.com/file/d/1KqgODtB9zxn3wutNk06DiaoHKMvaaVAU/view?usp=sharing), and place them in `./image_translation/pho_predictor/checkpoint`.

After run this script, we can get translated image, translated mask and material prediction material results, which would be written to a json. 

``````shell
python ./mateiral_transfer/transfer.py --name=rendered --dataset_mode=rendered --dataroot=./src/material_transfer/exemplar --gpu_ids=0 --batchSize=6 --use_attention --maskmix --noise_for_mask --warp_mask_losstype=direct --PONO --PONO_C --checkpoints_dir=./src/image_translation/checkpoint/preTrain --which_epoch=50 --save_per_img --nThreads=0 --no_pairing_check
``````

## Post-process

Based on generated mateiral json, we provide a python scripts to assgin mateirals by Blender. This script can generate blender files automaticly, and of course you can assign materials by other 3D processing softwares.

Please make sure that [materials](https://github.com/XiangyuSu611/TMT/tree/master/src/data_generation#materials) have been placed in `../../data/materials` before run this script.

``````shell
python ./material_transfer/assignment.py
``````

Once you get this blender file, you can view this 3D shape with high-quality materials and render it from any point of view👏. 