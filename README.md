# POTR-3D: 3D Multi-Person Pose Estimation from Monocular Videos

This repository contains the implementation of POTR-3D, a sequence-to-sequence 2D-to-3D lifting model designed for robust and smooth 3D multi-person pose estimation from monocular videos. The approach addresses challenges such as unseen views during training, occlusions, and output jittering.

## Features

- **Geometry-Aware Data Augmentation**: Generates diverse views while considering the ground plane and occlusions.
- **Robustness to Occlusions**: Effectively recovers poses even under heavy occlusions.
- **Smooth Outputs**: Produces natural and stable 3D pose estimations.

For more details, refer to the paper: [Towards Robust and Smooth 3D Multi-Person Pose Estimation from Monocular Videos in the Wild](https://arxiv.org/abs/2309.08644).

## Repository Structure

- `configs/`: Configuration files for training and evaluation.
- `data/`: Scripts and tools for data preprocessing.
- `lib/`: Core library containing model definitions and utilities.
- `run/`: Training and evaluation scripts.
- `tb/`: TensorBoard logs for monitoring training.

## Usage

1. Clone the repository
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. To train the model, run:
   ```bash
   bash potr3d.sh
   ```
For detailed instructions on data preparation and model evaluation, please refer to the respective scripts in the run/ directory.

---
## Citation
```
@article{park2023towards,
  title={Towards Robust and Smooth 3D Multi-Person Pose Estimation from Monocular Videos in the Wild},
  author={Park, Sungchan and You, Eunyi and Lee, Inhoe and Lee, Joonseok},
  journal={arXiv preprint arXiv:2309.08644},
  year={2023}
}
```
