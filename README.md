# S2LD: Sparse-to-Local-Dense Matching for Geometry-Guided Correspondence Estimation
### [Homepage](https://chenghao-li-1029.github.io/) | [Paper](https://ieeexplore.ieee.org/document/10159656)
<br/>

This is a reimplemented end-to-end version of S2LD correspondence estimation described in the original paper.

> Sparse-to-Local-Dense Matching for Geometry-Guided Correspondence Estimation
> [Shenghao Li](https://chenghao-li-1029.github.io), Qunfei Zhao<sup>\*</sup>, Zeyang Xia  
> IEEE Transaction on Image Processing, 2023

<div align="center">
  <img src="assets/s2ld.png" alt="demo_vid" width="400"/>
</div>

## Technical Architecture

S2LD proposes a novel sparse-to-local-dense matching framework for geometry-guided correspondence estimation. The architecture consists of three key components:

### 1. Attention-based Feature Extractor

- Utilizes attention mechanisms to extract features with global receptive fields
- Enables feature descriptors to capture global contextual information
- Enhances matching robustness and accuracy across different viewing conditions

### 2. Multi-level Matching Process

- **Sparse Matching Stage**: First establishes sparse correspondences across the entire image
- **Local Dense Matching Stage**: Performs dense matching in local regions around sparse keypoints
- Progressively refines correspondences at multiple levels to reduce reprojection errors
- Maintains sub-pixel level consistency while reducing computational complexity

### 3. 3D Noise-Aware Regularizer

- Designed through differentiable triangulation
- Provides additional 3D geometric guidance during training
- Handles supervision noise from camera pose and depth map errors
- Improves model generalization capability

## Key Innovations

### Asymmetric Sparse-to-Local-Dense Matching Strategy

Instead of performing dense matching across entire images (computationally expensive) or only sparse matching (less accurate), S2LD adopts an asymmetric approach:

- Efficiently detects sparse feature points with global context
- Densifies matches locally around geometrically promising regions
- Achieves the best trade-off between accuracy and efficiency

### Global Receptive Field with Attention

- Feature descriptors leverage attention mechanisms to achieve global receptive fields
- Captures long-range dependencies and contextual information
- Significantly improves matching robustness in challenging scenarios (occlusions, texture-poor regions)

### Geometry-Guided Correspondence Estimation

- Explicitly uses geometric information from sparse features to guide dense matching
- Reduces ambiguity in correspondence search space
- Ensures geometric consistency throughout the matching pipeline

### 3D-Aware Training with Noise Handling

- Novel 3D noise-aware regularizer handles imperfect supervision signals
- Differentiable triangulation provides 3D geometric constraints during training
- Improves robustness to camera pose and depth estimation errors

## Benchmark Results

S2LD demonstrates state-of-the-art performance on multiple geometric estimation benchmarks:

### MegaDepth-1500 Dataset

| Method | Type | AUC@5° | AUC@10° | AUC@20° | MMA@5E-4 |
|--------|------|--------|---------|---------|----------|
| Sup.+SG. | Sparse | 36.78% | 54.68% | 71.02% | **99.21%** |
| NCNet | Dense | 25.89% | 41.79% | 57.94% | 82.62% |
| DRCNet | Dense | 27.70% | 43.04% | 56.78% | 83.50% |
| LoFTR-DS | Dense | 48.41% | 65.04% | 78.28% | 95.43% |
| **S2LD (Ours)** | **Sparse-to-Dense** | **49.73%** | **65.69%** | **78.84%** | 96.16% |

**Key Achievements:**

- **+12.95%** improvement over best sparse method (Sup.+SG.) at AUC@5°
- **+1.32%** improvement over best dense method (LoFTR-DS) at AUC@5°
- Superior pose estimation accuracy with better computational efficiency

### Performance Highlights

- **Accuracy**: Achieves sub-pixel matching accuracy with high geometric consistency
- **Efficiency**: Faster than dense matching methods while more accurate than sparse methods
- **Robustness**: Handles challenging scenarios including occlusions, texture-poor regions, and large viewpoint changes
- **Generalization**: Strong performance across indoor and outdoor scenes

## Installation
```shell
# For full pytorch-lightning trainer features (recommended)
conda env create -f environment.yaml
conda activate s2ld
```

We provide the [download link](https://drive.google.com/drive/folders/1HwvPc3AzmjwxDsmBFIkWJmRdfF68_Hvw?usp=drive_link) to 
  - the megadepth-1500-testset.
  - The pretrained models of end2end S2LD.

## Run Demos

### Match image pairs

```shell
python demo_match.py --weight ./weights/s2ld-e2e-inference.pt
```

## Training

### Dataset Setup

Generally, MegaDepth is needed for training, the original dataset, the offline generated dataset indices. The dataset indices store scenes, image pairs, and other metadata within each dataset used for training/validation/testing. The relative poses between images used for training are directly cached in the indexing files. 

**Download the dataset indices**

You can download the required dataset indices from the [following link](https://drive.google.com/drive/folders/1HwvPc3AzmjwxDsmBFIkWJmRdfF68_Hvw?usp=drive_link).
After downloading, unzip the required files.
```shell
unzip downloaded-file.zip
# extract dataset indices
tar xf train-data/megadepth_indices.tar
# extract testing data (optional)
tar xf testdata/megadepth_test_1500.tar
```

**Build the dataset symlinks**

```shell
# megadepth
# -- # train and test dataset (train and test share the same dataset)
ln -s /path/to/megadepth/Undistorted_SfM /path/to/S2LD/data/megadepth/train
ln -s /path/to/megadepth/Undistorted_SfM /path/to/S2LD/data/megadepth/test
# -- # dataset indices
ln -s /path/to/megadepth_indices/* /path/to/S2LD/data/megadepth/index
```

### Training on MegaDepth

``` shell
scripts/train/train_outdoor_ds_e2e.sh
```

> NOTE: It uses 2 gpus only, with image sizes of 640x640. This is the reproduction of an end-to-end sparse-to-local-dense correspondence estimation, which may not be aligned to the results presented in the paper.

### Test on MegaDepth

``` shell
scripts/test/test_outdoor_ds_e2e.sh
```

For visualizing the results, please refer to `notebooks/visualize_dump_results.ipynb`.


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@ARTICLE{li2023s2ld,
  author={Li, Shenghao and Zhao, Qunfei and Xia, Zeyang},
  journal={IEEE Transactions on Image Processing}, 
  title={Sparse-to-Local-Dense Matching for Geometry-Guided Correspondence Estimation}, 
  year={2023},
  volume={32},
  number={},
  pages={3536-3551},
  doi={10.1109/TIP.2023.3287500}}
```