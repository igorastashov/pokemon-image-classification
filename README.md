## MLops project

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build Status](https://github.com/igorastashov/MLops-project/actions/workflows/checks.yml/badge.svg)](https://github.com/igorastashov/MLops-project/actions/workflows/checks.yml)

**Pokemon image classification using ConvNet model**

Astashov I.V., 2023.

This repository contains model, evaluation code and training code on dataset
from [kaggle](https://www.kaggle.com/datasets/lantian773030/pokemonclassification).
**If you would like to run pretrained model on your image see [(2) Quick start](https://github.com/igorastashov/MLops-project#2-quick-start)**.

**Objective**

The goal of this project is to develop a robust image classification pipeline for identifying Pokémon from images. This has applications in automated analysis and organization of Pokémon-related image datasets for both research and entertainment purposes.


## (1) Setup

### Install packages

- This template use poetry to manage dependencies of your project.
  First you need to [install poetry](https://python-poetry.org/docs/#installing-with-pipx);
- Next run `poetry install` and `poetry shell`.

## (2) Quick start

### Download model and optimizer

```
# Download model and optimizer
cd weights
bash download_weights.sh
cd ../..
```

### Run on a single image

This command runs the model on a single image, and outputs the prediction.
Put your Pokémon image into the appropriate folder `photo` to test.

```
# Model weights need to be downloaded
python infer.py
```

**Prediction and Deployment**

The model is integrated into a production pipeline for predictions. The pipeline consists of the following steps:

1. Preprocessing the input image (resize, normalize);
2. Feeding the image into the trained model;
3. Outputting the predicted Pokémon class.
   
For production use, the pipeline can be further extended to include batch processing and a web-based interface for user interaction. The infer.py script serves as the initial step for deploying predictions on single images. Future enhancements could involve integrating APIs and deploying the model on cloud platforms for scalability.

## (3) Dataset

### Download the dataset

A script for downloading the dataset is as follows:

```
# Download the dataset
cd data
bash download_data.sh
cd ../..
```

All the 150 Pokémon included in this dataset are from generation one.
There are around 25 - 50 images for each Pokémon.
All of them with the Pokémon in the center.
Most (if not all) of the images have relatively high quality (correct labels, centered).
The images don't have extremely high resolutions so it's perfect for some light deep learning.

Key challenges include:

- Limited dataset size, which may restrict model generalization;
- Potential variations in lighting, poses, and backgrounds across the dataset.
Despite these limitations, the dataset is sufficiently diverse for training a classification model.

If the script doesn't work, an alternative will be to download the zip files manually
from the [link](https://www.kaggle.com/datasets/lantian773030/pokemonclassification/download?datasetVersionNumber=1).
One can place the dataset zip files in `data`, respectively, and then unzip the zip file to set everything up.

**PAY ATTENTION**

**This repository runs Data Version Control (DVC) for training and validation data.
Pre-configured Google Drive remote storage stores raw input data.**

```console
$ dvc remote list
my_remote gdrive://1RXz3Mv7OxVveHtQ7c1ZtGgazDh6bPFJz
```

You can run `dvc pull` to download the data:

```console
$ dvc pull
```

## (4) Train and Evaluation model

Example script to train and evaluate model.

**Modeling Approach**

The project employs a custom-built convolutional neural network (ConvNet) designed specifically for the Pokémon classification task.

Key features:

- The ConvNet is implemented from scratch, without relying on pretrained weights;
- The architecture is optimized for the Pokémon dataset, balancing model complexity and performance to handle the relatively small dataset size effectively.

This approach ensures a tailored solution, focusing exclusively on the characteristics of the Pokémon images while maintaining flexibility for future modifications.

```
# Train ConvNet
python train.py
```

**Hydra - hyperparameter management tool.**

If you want to change hyper-parameters such as: `epoch_count`, `lr`, `batch_size`, `momentum`,
then you can change them in the file `conf/config.yaml`.

## (A) Acknowledgments

This repository borrows partially from [Isadrtdinov](https://github.com/isadrtdinov/intro-to-dl-hse/blob/2022-2023/seminars/201/seminar_04.ipynb), and [FUlyankin](https://github.com/FUlyankin/deep_learning_pytorch/tree/main/week08_fine_tuning) repositories.
Repository design taken from [v-goncharenko](https://github.com/v-goncharenko/data-science-template), [PeterWang512](https://github.com/PeterWang512/CNNDetection) and [ArjanCodes](https://github.com/ArjanCodes/2021-config).
