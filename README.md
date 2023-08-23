

### Object Detection setup

This code is developed based on pytorch 1.13.1, torchvision 0.14.1, (cudatoolkit 11.3.1 or 11.7) with either python 3.8.12 or python 3.10.9.

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:

```bash
conda create -n detector python=3.10.9
conda activate detector
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install easydict hydra-core=1.3.2

```
### Checkpoint file

Copy the checkpoint file to `outputs/checkpoints/ckpt_od_urchin_v00_clahe_r50fpn_156_base03`
### Execution
Update the variables at the start of the script to choose different images, score threshold, checkpoint, device etc.

```commandline
python source/single_image_inference.py
```
