# Material prediction
The material prediction network assigns materials from a collection of
realistic materials to the projected parts, based on the translated images and
perceptual similarity of the materials.

<div align=center><img src="https://github.com/XiangyuSu611/TMT/blob/master/docs/material_predicton_network.png" width="60%"></div>

## Get started
To train the material prediction network, we need to do following steps.

### **Step 1**: pre-compute material similarity distance matrix based on material database
```
python ./src/material_prediction/data/L2_LAB.py
python ./src/material_prediction/data/normalize_similarity_matrix.py
```

### **Step 2**: prepare network training data as lmdb files  
```
python ./src/material_prediction/data/make_lmdb.py
```

### **Step 3**: generate triplets for first-stage metric learning
```
python ./src/material_prediction/data/generate_triplet.py
```

### **Step 4**: metric learning
```
python ./src/material_prediction/train.py 
```

### **Step 5**: material classification learning
```
python ./src/material_prediction/train.py --classification --start-epoch=40 --epochs=220 --batch-size=180
```

