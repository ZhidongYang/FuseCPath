# FuseCPath
Official implementation of the paper: Fusion of Heterogeneous Pathology Foundation Models for Whole Slide Image Analysis.

## Introduction
Pathological FMs have exhibited substantial heterogeneity caused by diverse private training datasets and different network architectures. This heterogeneity introduces performance variability when we utilize the extracted features from different FMs in the downstream tasks. To fully explore the advantage of multiple FMs effectively, in this work, we propose a novel framework for the fusion of heterogeneous pathological FMs, called FuseCPath, yielding a model with a superior ensemble performance. The main contributions of our framework can be summarized as follows: <p align="right">
  <img src="assets/FuseCPath_logo.png" width="100">
</p> 
(i) To guarantee the representativeness of the training patches, we propose a multi-view clustering-based method to filter out the discriminative patches via multiple FMs' embeddings. (ii) To effectively fuse the features from heterogeneous patch-level FMs, we devise a cluster-level re-embedding strategy to online capture patch-level local features. (iii) To effectively fuse the slide-level FMs, we devise a collaborative distillation strategy to explore the connections between slide-level FMs. 

## System requirement
Ubuntu 20.04, CUDA version 12.0. <br>

## Major packages
```
python==3.10
torch==2.6.0
torchvision==0.21.0
huggingface-hub==0.30.2
openslide-python==1.4.2
trident
mvlearn
```

## Usage
```
python train.py --wsi_train_feat_dir ./training_emds \
 --model_output_path "./results" \
 --wsi_valid_feat_dir "./validation_emds" \
 --label_path "./labels" \
 --teacher_emds_path_1 "./teacher_1_emds" \
 --teacher_emds_path_2 "./teacher_2_emds" \
 --teacher_emds_path_3 "./teacher_3_emds" \
 --num_classes 2 \
 --epochs 60 \
 --batch-size 64 \
 --lr 0.0001 \
 --seed 2025
```

```
python predict.py --model_weight './results/model_epoch_30.pth' \
 --wsi_feat_dir './validation_emds' \
 --label_path './labels' \
 --emd_path './fused_emds'
```

## Acknowledgement

## Citation
