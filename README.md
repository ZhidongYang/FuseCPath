# FuseCPath
Official implementation of the paper: Fusion of Heterogeneous Pathology Foundation Models for Whole Slide Image Analysis.

## Introduction
Pathological FMs have exhibited substantial heterogeneity caused by diverse private training datasets and different network architectures. This heterogeneity introduces performance variability when we utilize the extracted features from different FMs in the downstream tasks. To fully explore the advantage of multiple FMs effectively, in this work, we propose a novel framework for the fusion of heterogeneous pathological FMs, called FuseCPath, yielding a model with a superior ensemble performance. The main contributions of our framework can be summarized as follows: (i) To guarantee the representativeness of the training patches, we propose a multi-view clustering-based method to filter out the discriminative patches via multiple FMs' embeddings. (ii) To effectively fuse the features from heterogeneous patch-level FMs, we devise a cluster-level re-embedding strategy to online capture patch-level local features. (iii) To effectively fuse the slide-level FMs, we devise a collaborative distillation strategy to explore the connections between slide-level FMs.

## System requirement
Ubuntu 20.04, CUDA version 12.0. <br>

## Major packages
python==3.10 <br>
torch==2.6.0 <br>
torchvision==0.21.0 <br>
huggingface-hub==0.30.2 <br>
openslide-python==1.4.2 <br>
trident <br>
mvlearn

## Usage
```
python train.py --wsi_train_feat_dir "path or list to training embeddings" \
 --model_output_path "path to saving the checkpoints" \
 --wsi_valid_feat_dir "path or list to validation embeddings" \
 --label_path "path or list to labels" \
 --teacher_emds_path_1 "path to slide FM 1 with its embeddings" \
 --teacher_emds_path_2 "path to slide FM 2 with its embeddings" \
 --teacher_emds_path_3 "path to slide FM 2 with its embeddings" \
 --num_classes 2 \
 --epochs 60 \
 --batch-size 64 \
 --lr 0.0001 \
 --seed 2025
```

## Acknowledgement

## Citation
