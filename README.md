<div align="center">
  
# SUGOcc: An Explicit Semantics and Uncertainty Guided Sparse Learning Framework for Real-Time 3D Occupancy Prediction 

</div>

## News
- [2026/01/21] The preprint version is available on [arXiv](https://arxiv.org/abs/2601.11396)

## Table of Contents
- [Method](#method)
- [Getting Start](#getting-start)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Method
![Overview](images/pipeline.png)
<center> Overview of the proposed SUG-Occ framework.</center>

## Getting Start (Coming Soon)
### Installation
```python
## Basic Installation
conda create -n sugocc python=3.8 -y
conda activate sugocc
conda install gcc==9.4 gxx==9.4  ## conda compiler need to relink ld, otherwise use system compiler
conda install cudatoolkit==11.7 cudatoolkit-dev==11.7 -c conda-forge -y
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

## MMLab Framwork
pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmdet3d==1.4.0

## Minkowski Engine
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --global-option="--blas_include_dirs=${CONDA_PREFIX}/include" --global-option="--blas=openblas"

## Occ Pool CUDA Operation
python setup.py develop
```
### Data Preparation
#### SemanticKITTI
Download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color, velodyne laser data, and calibration files) and the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download). Put all `.zip` files under `SparseOcc/data/SemanticKITTI` and unzip these files. Then you should get the following dataset structure:
```
SUGOcc
├── data/
│   ├── SemanticKITTI/
│   │   ├── dataset/
│   │   │   ├── sequences
│   │   │   │   ├── 00
│   │   │   │   │   ├── calib.txt
│   │   │   │   │   ├── image_2/
│   │   │   │   │   ├── image_3/
│   │   │   │   │   ├── voxels/
│   │   │   │   ├── 01
│   │   │   │   ├── 02
│   │   │   │   ├── ...
│   │   │   │   ├── 21
```
Preprocess the annotations for semantic scene completion:
```bash
python projects/SUGOcc/sugocc/tools/kitti_process/semantic_kitti_preprocess.py --kitti_root data/SemanticKITTI --kitti_preprocess_root data/SemanticKITTI --data_info_path projects/SUGOcc/sugocc/tools/kitti_process/semantic-kitti.yaml
```
#### Occ3D-Nuscenes
Download the full nuScenes dataset from the [official website](https://www.nuscenes.org/download).

Download the primary Occ3D-NuScenes annotations from the [project page](https://tsinghua-mars-lab.github.io/Occ3D/).

Organize your data following this structure:

```
SUGOcc/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── gts/                 # Main Occ3D annotations
```
Run the preprocessing scripts to prepare the data for training:
```
PYTHONPATH=$(pwd):$PYTHONPATH python tools/create_data_bevdet.py
```
### Pre-trained Models
Download the pre-trained image encoder from [MaskDINO](https://github.com/IDEA-Research/MaskDINO) (maskdino_r50_50e_300q_panoptic_pq53.0.pth)
### Train & Evaluate
#### Training
```python
bash tools/dist_train.sh projects/SUGOcc/configs/$CONFIG $GPUS
```
#### Evaluation
```python
bash tools/dist_test.sh projects/SUGOcc/configs/$CONFIG $PTH 1
```
## Results
#### Results on SemanticKITTI

| Model | Backbone | mIoU | FPS | Config | Weights|
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| SUGOcc| ResNet-50 | 14.91 | 10.1 | [config](projects/SUGOcc/configs/sugocc_kitti.py) | [weights](https://github.com/tlab-wide/SUGOcc/releases/download/V1.0/sugocc_kitti.pth) |


#### Results on Occ3D-Nuscenes
coming soon

## Citation
If you find our work useful for your research, please consider citing the paper:
```bash
@article{wu2026sug,
  title={SUG-Occ: An Explicit Semantics and Uncertainty Guided Sparse Learning Framework for Real-Time 3D Occupancy Prediction},
  author={Wu, Hanlin and Lin, Pengfei and Javanmardi, Ehsan and Bao, Naren and Qian, Bo and Si, Hao and Tsukada, Manabu},
  journal={arXiv preprint arXiv:2601.11396},
  year={2026}
}
```
## Acknowledgements
Many thanks to these excellent projects:
- [OccFormer](https://github.com/zhangyp15/OccFormer)
- [SparseOcc](https://github.com/VISION-SJTU/SparseOcc)
- [ProtoOcc](https://github.com/SPA-junghokim/ProtoOcc)
- [ALOcc](https://github.com/cdb342/ALOcc)
- [MaskDINO](https://github.com/IDEA-Research/MaskDINO)
