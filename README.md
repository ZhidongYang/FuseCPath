# FuseCPath
Official implementation of the paper: Fusion of Heterogeneous Pathology Foundation Models for Whole Slide Image Analysis.

## Introduction
Pathological FMs have exhibited substantial heterogeneity caused by diverse private training datasets and different network architectures. This heterogeneity introduces performance variability when we utilize the extracted features from different FMs in the downstream tasks. To fully explore the advantage of multiple FMs effectively, in this work, we propose a novel framework for the fusion of heterogeneous pathological FMs, called FuseCPath, yielding a model with a superior ensemble performance. The main contributions of our framework can be summarized as follows: (i) To guarantee the representativeness of the training patches, we propose a multi-view clustering-based method to filter out the discriminative patches via multiple FMs' embeddings. (ii) To effectively fuse the features from heterogeneous patch-level FMs, we devise a cluster-level re-embedding strategy to online capture patch-level local features. (iii) To effectively fuse the slide-level FMs, we devise a collaborative distillation strategy to explore the connections between slide-level FMs.

## System requirement
Ubuntu 20.04, CUDA version 12.0. <br>

## Required packages

## Usage

## Acknowledgement

## Citation
