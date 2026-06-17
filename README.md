# PC-TGS: Point-Cloud-Assistant Localized Statistical Channel Prediction by Tangent Gaussian Splatting

[Ye Xue](https://yokoxue.github.io/), [Yiheng Wang](https://yeehengwang.github.io/), Xinhua Shao, Qi Yan, Shutao Zhang, [Tsung-Hui Chang](https://myweb.cuhk.edu.cn/changtsunghui/Home)

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://doi.org/10.1109/TWC.2026.3696997)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/yeehengwang/HUAWEI-RF-3DGS-Project)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://chenjiadie.github.io/PC-TGS/)

![PC-TGS Framework](flow_00.png)

PC-TGS is the first framework to **extrapolate** channel angular power spectrum (APS) to unmeasured outdoor locations by integrating sparse RSRP measurements with dense LiDAR geometry. Published at **IEEE Transactions on Wireless Communications, 2026**.

For details, full performance comparisons, and BibTeX, see the [Project Page](https://chenjiadie.github.io/PC-TGS/).

---

## Quick Start

### Environment

```bash
# Python >= 3.8, CUDA-enabled GPU
pip install numpy scipy h5py pyyaml tqdm einops matplotlib
pip install torch torchvision torchaudio  # match your CUDA version
```

> **Note:** The training dataset is proprietary. To use this code with your own data, prepare files matching the format expected by `datasets_aps_new.py`.

### Training

```bash
python train_radsplatter_new.py \
  --config ./radsplatter_setting_new.yml \
  --gpu 0 \
  --mode train \
  --num_scatters 2000 \
  --world_size 1 \
  --num_max_angles 800 \
  --sh_up_iter 500
```

### Evaluation

```bash
python train_radsplatter_new.py \
  --config ./radsplatter_setting_new.yml \
  --gpu 0 \
  --mode test \
  --num_scatters 2000 \
  --world_size 1 \
  --num_max_angles 800
```

---

## Repository Structure

```text
├── README.md
├── radsplatter_setting_new.yml         # Training and optimizer configuration
├── train_radsplatter_new.py            # Main training/testing entrance
├── radsplatter_model.py                # PC-TGS model (RM, SH coefficients, scatterer attributes)
├── radsplatter_render.py               # Tangent-plane projection + electromagnetic splatting
├── datasets_aps_new.py                 # Dataset loaders for RSRP and APS data
├── projection_utils.py                 # 3D-to-angular projection + Jacobian computation
├── complex_sh_utils_new.py             # Complex spherical harmonic evaluation
├── sh_utils.py                         # Real spherical harmonic utilities
├── pdf_utils.py                        # Gaussian PDF computation
├── prune_utils.py                      # Mahalanobis-based Gaussian filtering
├── loss_utils.py                       # Loss functions (L1, L2, SSIM, SmoothL1)
├── data_painter.py                     # APS visualization and data processing
└── utils.py                            # General tensor and rotation utilities
```

---

## Citation

If you find this work helpful, please cite:

```bibtex
@article{xue2026point,
  title={Point-Cloud-Assistant Localized Statistical Channel Prediction by Tangent Gaussian Splatting},
  author={Xue, Ye and Wang, Yiheng and Shao, Xinhua and Yan, Qi and Zhang, Shutao and Chang, Tsung-Hui},
  journal={IEEE Transactions on Wireless Communications},
  volume={25},
  pages={17816--17830},
  year={2026},
  publisher={IEEE},
  doi={10.1109/TWC.2026.3696997}
}
```

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
