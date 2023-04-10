# CLIP4MC: An RL-Friendly Vision-Language Model for Minecraft
<div align="center">

[[Website]](https://sites.google.com/view/clip4mc) 
[[Arxiv Paper]](https://arxiv.org/pdf/2303.10571)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/MineDojo)](https://pypi.org/project/MineDojo/)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/MineDojo/MineCLIP)](https://github.com/PKU-RL/Plan4MC/blob/main/LICENSE)
______________________________________________________________________
![](figs/arch.png)
</div>

CLIP4MC is a Vision-Language model for Minecraft, aligning actions implicitly contained in the video and transcript clips in addition to entities. We construct and release a neat vision-language dataset for Minecraft based on YouTube datset from [MineDojo](https://github.com/MineDojo/MineDojo), and we train our CLIP4MC model on the constructed dataset. Empirically, our method can provide a more friendly reward signal for the RL training procedure.

## Demonstration  
Here are some demonstrations of agent trained with CLIP4MC.

| Harvest a leaf | Milk a cow | Shear a sheep |
| :---: | :---: | :---: |
|<img src="figs/harvest_a_leaf.gif" width="200"/>|<img src="figs/milk_a_cow.gif" width="200"/>|<img src="figs/shear_a_sheep.gif" width="200"/>|

## Installation

- Install python packages in `requirements.txt`. Note that we require PyTorch>=1.9.0 and x-transformers==0.27.1.

- Dataset should get ready before training. Information of each data piece is available in our released [dataset](https://drive.google.com/drive/folders/19vDy2jaooF74MDt3dLAsyLRpRcUFKVCY?usp=sharing).

- A ViT-16 version of pretrained CLIP is needed.

## Run for train

```
torchrun --nproc_per_node=4 train_ddp.py --dataset_log_file XXX --pretrain_model_path XXX --model_type CLIP4MC | CLIP4MC_simple | MineCLIP
```

## Citation
```bibtex
@article{ding2023clip4mc,
  title={CLIP4MC: An RL-Friendly Vision-Language Model for Minecraft},
  author={Ding, Ziluo and Luo, Hao and Li, Ke and Yue, Junpeng and Huang, Tiejun and Lu, Zongqing},
  journal={arXiv preprint arXiv:2303.10571},
  year={2023}
}
```


