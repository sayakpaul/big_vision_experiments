# Experiments with `big_vision`

Contains my experiments with the [`big_vision`](https://github.com/google-research/big_vision) repository to train ViTs on ImageNet-1k.

## What is `big_vision`?

From the repository:

> This codebase is designed for training large-scale vision models on Cloud TPU VMs. It is based on Jax/Flax libraries, and uses tf.data and TensorFlow Datasets for scalable input pipelines in the Cloud.

> big_vision aims to support research projects at Google. We are unlikely to work on feature requests or accept external contributions, unless they were pre-approved (ask in an issue first). 

## Why this repository?

* I really like how `big_vision` is organized into composable modules.
* I wanted to reproduce some of the ImageNet-1k results reported by the `big_vision` authors.
* `big_vision` not only reports scores for ImageNet-1k validation set but also reports
scores for ImageNet-V2 and ImageNet-Real.
* I wanted to run the entire training using Cloud TPUs and the same time I wanted to 
improve by JAX skills.
* For the sheer joy of training models to SoTA.

This repository will also contain the trained checkpoints and the training logs. Additionally, 
this Colab Notebook () takes the raw training logs and generates a plot for reporting accuracies
across three benchmarks: ImageNet-1k validation set, ImageNetV2, ImageNet-Real.

[TBA PLOT]

## Setup

Even though the `big_vision` repository provides instructions for setting things up I found them a bit incomplete.
Hence, I developed another one. Find it here - [`setup.md`](https://github.com/sayakpaul/big_vision_experiments/blob/main/setup.md).
