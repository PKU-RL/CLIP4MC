# Reinforcement Learning Friendly Vision-Language Model for MineCraft

<div align="center">
[[ECCV 2024]](https://arxiv.org/pdf/2303.10571)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/MineDojo)](https://pypi.org/project/MineDojo/)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/MineDojo/MineCLIP)](https://github.com/PKU-RL/Plan4MC/blob/main/LICENSE)
</div>

We propose a novel cross-modal contrastive learning framework architecture, CLIP4MC, aiming to learn a reinforcement learning (RL) friendly vision-language model (VLM) that serves as an intrinsic reward function for open-ended tasks. Simply utilizing the similarity between the video snippet and the language prompt is not RL-friendly since standard VLMs may only capture the similarity at a coarse level. To achieve RL-friendliness, we incorporate the **task completion degree** into the VLM training objective, as this information can assist agents in distinguishing the importance between different states. 

<img src="figs/arch.png" alt="arch" style="zoom:30%;" />

Moreover, we provide **neat YouTube datasets** based on the large-scale YouTube database provided by MineDojo. Specifically, two rounds of filtering operations guarantee that the dataset covers enough essential information and that the video-text pair is highly correlated. Empirically, we demonstrate that the proposed method achieves better performance on RL tasks compared with baselines.

## Packages
Install python packages in `requirements.txt`.

Note that we require `PyTorch>=1.10.0` and `x-transformers==0.27.1`.

## Data
Dataset should get ready before training. Information of each data piece is available in our released [dataset](https://huggingface.co/datasets/AnonymousUserCLIP4MC/CLIP4MC).

In this project we provide a naive implementation of dataloader and dataset. To use the dataloader and dataset, the data should be organized in the following structure:

```
data_dir_0
├── text_input.pkl
├── video_input.pkl
├── size.json
data_dir_1
├── text_input.pkl
├── video_input.pkl
├── size.json
...
data_dir_n
├── text_input.pkl
├── video_input.pkl
├── size.json
```
Use the tokenizer corresponding to the clip in AutoTokenizer to tokenize the natural language in the released [dataset](https://huggingface.co/datasets/AnonymousUserCLIP4MC/CLIP4MC) and save it as a pickle file. Convert the video clip with the corresponding timestamp into a pickle file.

A log file for each dataset is also required. The log file should be a `json` file with the following structure:
  ```
  {
    "train": [data_dir_0, data_dir_1, ..., data_dir_n],
    "test" : [data_dir_0, data_dir_1, ..., data_dir_n],
  }
  ```
The `train` and `test` keys are required. The `train` key should contain a list of data directories for training. The `test` key should contain a list of data directories for testing.

For the split of training and test sets, please refer to our released [dataset](https://huggingface.co/datasets/AnonymousUserCLIP4MC/CLIP4MC).

The log file should be filled in the function `get_processed_list` in `process/static.py`.

### Pretrained Models

- A ViT-B/16 version of pretrained CLIP is required for training from scratch. You can download it from [here](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt).

Please fill in the path of the downloaded ViT-B-16 CLIP weight into the `--pretrain_model_path` of `train_ddp_clip4mc.py` and `train_ddp_mineclip.py`.

## Usage

You can use the scripts below to train the model.

* `run_clip4mc.sh` is used to run the training process of CLIP4MC.
* `run_mineclip.sh` is used to run the training process of MineCLIP.

## Citation

If you find our work useful in your research and would like to cite our project, please use the following citation:

```latex
@article{jiang2024reinforcement,
 title={Reinforcement Learning Friendly Vision-Language Model for Minecraft},
 author={Jiang, Haobin and Yue, Junpeng and Luo, Hao and Ding, Ziluo and Lu, Zongqing},
 journal={arXiv preprint arXiv:2303.10571},
 year={2024}
}
```

