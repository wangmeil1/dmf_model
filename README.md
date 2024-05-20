This is a [PyTorch](https://pytorch.org/) implementation of Deep Multi-Model Fusion for Single-Image Dehazing(https://ieeexplore.ieee.org/document/9009514) and its improved model.
### Instructions
- Make sure you have Python>=3.7 and pytorch>=2.0 installed. 

- 16 GB or larger GPU memory is recommended.

### DATASET
- Download the RESIDE dataset from (https://sites.google.com/site/boyilics/website-builder/reside).

- Download the O-Haze dataset from (https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/).

- Make a directory ```./data``` and create a symbolic link for uncompressed data, e.g., ```./data/RESIDE```.

- The self-selected pictures are also stored in the ```./data``` directory, e.g., ```./data/test_p```.

### TRAINING
- Set the path of datasets in ```tools/config.py```.
- run ```python train.py``` to train the baseline model on the RESIDE dataset.
- run ```python new_train.py``` to train the improved model on the RESIDE dataset.
- run ```python train_ohaze.py``` to train the baseline model on the O-Haze dataset.
- run ```python new_train_ohaze.py``` to train the improved model on the O-Haze dataset.
- The baseline model code is in ```model.py```, and the improved model code is in ```new_model.py```.

### TESTING 
- Set the path of five benchmark datasets in ```tools/config.py```.
- Put the trained model in ```./ckpt/```.
- test the baseline model by running ```python test.py```.
- test the improved model by running ```python new_test.py```.
- test and show results of self-selected pictures by running ```python test_on_test_p.py```.

 **Model parameters and self-selected pictures can be downloaded from baidu netdisk.**
 链接：(https://pan.baidu.com/s/10uBRH_S78P0hpQyiTxosHA?pwd=t7ju) 
提取码：t7ju
