# Material prediction
The material prediction network assigns materials from a collection of
realistic materials to the projected parts, based on the translated images and
perceptual similarity of the materials.

<div align=center><img src="https://github.com/XiangyuSu611/TMT/blob/master/docs/material_predicton_network.png" width="60%"></div>

## Get started
To train the material prediction network, we nedd to do following steps.

### **step 1**: pre-compute material similarity distance matrix based on material database
```
python -m src.material_prediction.data.L2_LAB
python -m src.material_prediction.data.normalize_similarity_matrix
```

### **step 2**: prepare network training data as lmdb files  
```
python -m src.material_prediction.data.make_lmdb
```

### **step 3**: generate triplets for first-stage metric learning
```
python -m src.material_prediction.data.generate_triplet
```

### **step 4**: metric learning
```
python -m src.material_prediction.train 
```

### **step 5**: material classification learning
```
python -m src.material_prediction.train --classification --start-epoch=40 --epochs=220 --batch-size=180
```