

### Object Detection setup

This code is developed based on pytorch 1.13.1, torchvision 0.14.1, (cudatoolkit 11.3.1 or 11.7) with either python 3.8.12 or python 3.10.9.

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:

```bash
conda create -n detector python=3.10.9
conda activate detector
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib=3.7.0
conda install easydict hydra-core=1.3.2
pip install opencv-python==4.7.0.72 Pygments
```

maybe also 
```bash
conda install -c conda-forge  pandas scikit-learn seaborn
conda install easydict hydra-core=1.3.2 parse pick pycocotools prompt_toolkit
```