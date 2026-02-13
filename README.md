# Incremental-Training-Protein
This is repository for Incremrental learning.

This has following main python files.
  1) Model.py - This python file contains the code for CNN-based segment encoder.
  2) Incremental_Training.py - This contains the code for segmentation + incremental training.
  3) Predict.py - This contains the code for the predicting on unseen data.
  4) Evaluate.py - This contains the code for the evaluation metrics.

This repository contains the dataset for the BP and MF gene ontology.

# Trainig from scratch
The major steps with training are as follows:

Step 1: Run `python Model.py`                   <==== Create an instance of LiteSeqCNN

Step 2: Run `python Incremental_Training.py`    <==== Start the incremental training

# Trainig from scratch
The major steps with testing are as follows:

Step 1: Run `python Evaluate.py`                <==== Code for the evaluation metrics.

Step 2: Run `python Predict.py`                 <==== Code for performing the prediction on unseen data.
