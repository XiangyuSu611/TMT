# Material transfer

Use pre-trained image translation and material prediction network, we can get final material transfer results in this part.

## Pre-process input

Our pipeline starts with a random wild photo and a 3D shape with semantic segmentation.

For wild photo, we should remove background and estimate camera pose. There are many ways to remove the background, if you don't want to configure addtional environment, we recommend [remove.bg](https://www.remove.bg/). And we recommend to move object to the center of image for better results.

````shell
python ./material_transfer/pre_process/crop_and_center.py
````

To estimate the camera pose, we use pre-trained [camera pose estimator](https://github.com/laughtervv/DISN),  please download this [model](), place in `./cam_estimation/checkpoint/`, and run this script.

````shell
python ./material_transfer/cam_estimation/est_cam_syn.py --cam_est --cam_log_dir /home/code/TMT/src/material_transfer/cam_estimation/checkpoint/
````

Next, we projection 3D shape from estimated camera pose to get semantic projection, and generate test pairs.

``````shell
python ./material_transfer/pre_process/generate_test_pair.py
``````

## Run network

Please download our pre-trained models, and place them in `./models`.

After run this script, we can get translated image, translated mask and material prediction results, which would be written to a json. 

``````shell
python ./mateiral_transfer/transfer.py --name=rendered --dataset_mode=rendered --dataroot=./src/material_transfer/exemplar --gpu_ids=0 --batchSize=6 --use_attention --maskmix --noise_for_mask --warp_mask_losstype=direct --PONO --PONO_C --checkpoints_dir=./src/image_translation/checkpoint/preTrain --which_epoch=50 --save_per_img --nThreads=0 --no_pairing_check
``````

## Post-process output

Based on generated mateiral json, we provide a python scripts to assgin mateirals by Blender. This script can generate blender files automaticly, and of course you can assign materials by other 3D processing softwares.

``````shell
python ./material_transfer/assignment.py
``````

Once you get this blender file, you can view this 3D shape with high-quality materials and render it from any point of view👏. 