# TMT: Translation-based Material Transfer
![image](https://github.com/XiangyuSu611/TMT/blob/master/docs/teaser.png)
This is the author's code release for:

> **Photo-to-shape Material Transfer for Diverse Structures**  
> Ruizhen Hu, Xiangyu Su, Xiangkai Chen, Oliver Van Kaick, Hui Huang.  
> **ACM Trans. on Graphics (Proc. SIGGRAPH). 41(4), 2022.**

##  Introduction
We introduce a method for assigning photorealistic relightable materials
to 3D shapes in an automatic manner. Our method takes as input a photo
exemplar of a real object and a 3D object with segmentation, and uses the
exemplar to guide the assignment of materials to the parts of the shape, so
that the appearance of the resulting shape is as similar as possible to the
exemplar. 

For more details and materials, please refer to our [[project page](https://vcc.tech/research/2022/TMT)].


![image](https://github.com/XiangyuSu611/TMT/blob/master/docs/overview.png)

## Getting started
For all users, please read **dependency** for initial environment setting. Then, for users who want to just use our pre-trained model, please read **material translation**. And for users who want to re-train models with their own data, please read **data pre-process** for more help. 

Enjoy it!😎

* [Dependency](url)
* [Data pre-process](url)
* [Material translation](url)

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