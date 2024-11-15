<div align="center">

# CrowdMAC: Masked Crowd Density Completion for Robust Crowd Density Forecasting (WACV'25)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

**[[Paper](https://arxiv.org/abs/2407.14725)]**

</div>

This is the official code release for our WACV 2024 paper \
"CrowdMAC: Masked Crowd Density Completion for Robust Crowd Density Forecasting".

## 🔨 Installation
```bash
pip install -r requirements.txt
```

</div>

## 🔥 Training
### 1. Pretraining
```bash
python train.py
```

### 2. Fine-tuning
Place the [sdd-pretrained.pth](https://keio.box.com/s/enoy1nr1pmm57dub6tr6vj72mi9qa6tp) under pretrained folder.
```bash
bash finetune.sh
```

## 👏 Acknowledgement

We sincerely thank the authors of [VideoMAE](https://github.com/MCG-NJU/VideoMAE) for providing their source code, which has been invaluable to our work. We are immensely grateful for their contribution.



## ✍️ Citation
If you use this code for your research, please cite our paper.
```bib
@inproceedings{FUJII2025CrowdMAC,
    author = {Ryo Fujii, Ryo Hachiuma, and Hideo Saito},
    title = {CrowdMAC: Masked Crowd Density Completion for Robust Crowd Density Forecasting},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year = {2025},
}
```