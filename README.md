# Incremental-Training-Protein
This is repository for the Incremental learning.

This has following main python files.
  1) Model.py - This python file contains the code for CNN-based segment encoder.
  2) Incremental_Training.py - This contains the code for segmentation + incremental training.
  3) Predict.py - This contains the code for the predicting on unseen data.
  4) Evaluate.py - This contains the code for the evaluation metrics.

This repository contains the dataset for the BP and MF gene ontology.

# Requirements
1. Python
2. Tensorflow >= 2.0
3. Keras
4. Pandas, Numpy

# Dataset
The dataset for BP and MF is presents in the BP and MF folders, respectively.

# Trainig from scratch
Download the BP and MF datasets and set the training path in `Model.py` and `Incremental.py`.
The major steps with training are as follows:

Step 1: Run `python Model.py`                   <==== Create an instance of LiteSeqCNN

Step 2: Run `python Incremental_Training.py`    <==== Start the incremental training

# Testing
The major steps with testing are as follows:

Step 1: Run `python Evaluate.py`                <==== Code for the evaluation metrics.

Step 2: Run `python Predict.py`                 <==== Code for performing the prediction on unseen data.

The paper is under review at Scientific Reports.
