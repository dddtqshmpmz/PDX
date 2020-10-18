# PDX

- [Overview](#overview)
- [Software Requirements](#software-requirements)
- [Installation Guide](#installation-guide)
- [How to Run](#how-to-run)
- [License](#license)

## Overview

This  is a project which establishes an optimal predictive model to screen lung cancer patients for NOG/PDX models, and also offers a general approach for building prediction models in small unbalanced biomedical samples based on machine learning.

## Software Requirements

### OS Requirements

This project is supported for Linux and Windows, and it has been tested on the following systems:

+ Linux: Ubuntu 16.04
+ Windows 10

### Python Dependencies

The code is compatible with Python 3.7. The following dependencies are needed to run the training or testing tasks:

```python
numpy
pandas
sklearn
smote-variants
xgboost
catboost
```

## Installation Guide

You can get our basic codes from Github.

```
git clone https://github.com/dddtqshmpmz/PDX.git
```

## How to Run

1. Install all the  dependencies.
   + `pip install -r requirements.txt`
2. Pre-process data.
   + We randomly resample the data from the original dataset (`original_data.csv`) and generate 100 datasets which are saved in `/tmpData100` directory. We do not provide the datasets due to data privacy.
3. Use different machine learning methods (CatBoost, XGBoost, SVM and LR) to train the PDX prediction models.
   + `python classify_with_smote.py`  Train and test different models with or without SMOTE. The mean scores (AUC, precision, recall, accuracy and F1-score) of different models are saved in `/score` and `/scoreWithDifferentFeatures` directories.
   + `python cross_validation.py`  Use K-fold cross validation to get classification scores of train/val/test datasets.
   + `python ensemble_learning.py`  Integrate multiple models to get final test scores.
   + You can see some results based on different models with different feature selections in `/scoreWithDifferentFeatures` directory.

## License

This project is covered under the **Apache 2.0 License**.