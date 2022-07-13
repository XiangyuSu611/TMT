# TMT: Translation-based Material Transfer
![image](https://github.com/XiangyuSu611/TMT/blob/master/docs/teaser.png)
**This is the author's code release for:**
> **Photo-to-shape Material Transfer for Diverse Structures**  
> Ruizhen Hu, Xiangyu Su, Xiangkai Chen, Oliver Van Kaick, Hui Huang.  
> **ACM Trans. on Graphics (Proc. SIGGRAPH). 41(4), 2022.**

##  Introduction
We introduce a **T**ranslation-based **M**aterial **T**ransfer **(TMT)** method for assigning photorealistic relightable materials to 3D shapes in an automatic manner. Our method takes as input a photo exemplar of a real object and a 3D object with segmentation, and uses the exemplar to guide the assignment of materials to the parts of the shape, so that the appearance of the resulting shape is as similar as possible to the exemplar. To accomplish this goal, our method combines an **image translation** neural network with a **material assignment** neural network. The image translation network translates the color from the exemplar to a projection of the 3D shape and the part segmentation from the projection to the exemplar. Then, the material prediction network assigns materials from a collection of realistic materials to the projected parts, based on the translated images and perceptual similarity of the materials.


For more details and materials, please refer to our [project page](https://vcc.tech/research/2022/TMT).


![image](https://github.com/XiangyuSu611/TMT/blob/master/docs/overview.png)

## Getting started
First, please read **Environment** for initial environment setting. Then, for users who want to just use our pre-trained model, please go to **Material transfer**. And for users who want to re-train models with their own data, please read **Data generation, image translation, and mateiral prediction** for more help. 

😎Enjoy it!

* [Environment](https://github.com/XiangyuSu611/TMT/blob/master/docs/Environment.md)
* [Data generation](https://github.com/XiangyuSu611/TMT/tree/master/src/data_generation)
* [Image translaion](https://github.com/XiangyuSu611/TMT/blob/master/src/image_translation)
* [Material prediction](https://github.com/XiangyuSu611/TMT/blob/master/src/material_prediction)
* [Material transfer](https://github.com/XiangyuSu611/TMT/blob/master/src/material_transfer)

## Citation

Please cite this paper in your publications if it helps your research:

```bibtex
@article{TMT,
title = {Photo-to-Shape Material Transfer for Diverse Structures},
author = {Ruizhen Hu and Xiangyu Su and Xiangkai Chen and Oliver van Kaick and Hui Huang},
journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH)},
volume = {39},
number = {6},
pages = {113:1--113:14},
year = {2022},
}
```

## License

Our code is released under MIT License. See LICENSE file for details.