# Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation

<a href='https://nautilusmeshgen.github.io'><img src='https://img.shields.io/badge/Project-Page-blue'></a> 
<a href='https://arxiv.org/abs/2501.14317'><img src='https://img.shields.io/badge/arXiv-2501.14317-b31b1b.svg'></a> 
   

<!-- ### [Project Page](https://nautilusmeshgen.github.io) | [Paper](https://arxiv.org/abs/2501.14317) -->

Yuxuan Wang*, Xuanyu Yi*, Haohan Weng*, Qingshan Xu, Xiaokang Wei, 

Xianghui Yang, Chunchao Guo, Long Chen, Hanwang Zhang

(_*Equal Contribution_)

Nanyang Technological University, Tencent Hunyuan, 

The Hong Kong Polytechnic University, Hong Kong University of Science and Technology

![image](https://github.com/Yuxuan-W/Nautilus/blob/main/figures/representative_img.jpg)

## Preparation

### Trained Model

**We plan to release the checkpoints upon the acceptance of the paper.**

Due to company confidentiality policies,
we are unable to release the model trained on the full dataset with the 1024-dimension Michelangelo encoder.
Instead, we provide a version trained with the 256-dimension Michelangelo encoder and a filtered dataset solely based on Objaverse.
The performance of this model is moderately lower compared to the full version.

### Installation

Install the packages in `requirements.txt`. The code is tested under CUDA version 11.8.

```bash
# clone the repository
git clone https://github.com/Yuxuan-W/nautilus.git
cd nautilus
# create a new conda environment
conda create -n nautilus python=3.9 -y
conda activate nautilus
# install pytorch, we use cuda 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia -y
# install other dependencies
pip install -r requirements.txt
# install torch-scatter and torch-cluster based on your cuda version
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

## Inference

The generation inference will be available upon the release of our checkpoints.
Generating a 5000-face mesh asset typically takes around 3 to 4 minutes on a single A100 GPU.

```bash
bash infer.sh /your/path/to/checkpoint /your/path/to/pointcloud
```

## Citation

If you find our work helps, please cite our paper:
```
@article{wang2025nautilus,
  title={Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation},
  author={Wang, Yuxuan and Yi, Xuanyu and Weng, Haohan and Xu, Qingshan and Wei, Xiaokang and Yang, Xianghui and Guo, Chunchao and Chen, Long and Zhang, Hanwang},
  journal={arXiv preprint arXiv:2501.14317},
  year={2025}
}
```

<br/>

## Acknowledgement
Our code is based on the wonderful repo of [meshgpt-pytorch](https://github.com/lucidrains/meshgpt-pytorch). 
We thank the authors for their great work.

