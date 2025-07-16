# Higgs boson pT prediction

This repository contains code for reconstructing the transverse momentum (pT) of the Higgs boson in H → tau+ tau− decays using a deep neural network. The project was developed as part of a bachelor's thesis at the Faculty of Mathematics and Physics, Charles University.

## #Dataset

**DatasetClass.py**  
Defines a class for loading and preprocessing Monte Carlo simulated data.
- Loads ROOT files with tau decay and jet information
- Selects relevant kinematic variables (e.g. tau momenta, dijet system, missing transverse energy)
- Decomposes 4-momenta into transverse momentum, pseudorapidity, azimuthal angle and mass
- Outputs TensorFlow-ready dataset objects with optional weighting and shuffling

**data_mmc**  
Dataset including the ditau_mmc_mlm_m feature.

**data_d**  
Same dataset as above, but without the MMC-based feature.

## #Model

**ModelClass.py**  
Implements the regression model architecture using TensorFlow Keras.
- Deep residual MLP with configurable number of blocks and neurons
- Layers: Dense → BatchNorm → ReLU → Dropout
- Includes forward pass, training loop, validation, and early stopping
- Logs evaluation metrics such as mean squared error and mean absolute percentage error

## #Helpers

**helpers.py**  
Utility functions used throughout the project.
- MAPE and residual computation
- Loss plotting and histogram generation
- Logging to file
- Argument parsing helpers

## #Main

**main.py**  
Main executable script that trains the model.
- Loads dataset via DatasetClass
- Builds model using ModelClass
- Parses command-line arguments (learning rate, batch size, epochs, etc.)
- Runs training, evaluation, and saves model and logs

## #Notebooks

**higgs_pt.ipynb**, **higgs_pt_prediction.ipynb**, **predict.ipynb**  
Jupyter notebooks used for generating and visualizing plots that appear in the thesis.
- Plotting residual distributions
- Comparing predictions vs. targets
- Visualizing model performance

## #Experiments and Results

**main_optuna.py**  
Performs hyperparameter optimization using the Optuna library.

**model_comparsion.txt**  
Text summary comparing different model variants, especially the effect of removing the MMC feature and Dropout layer.

**train_history_pt_mmc.txt**  
Raw training history (loss and metric values) for different model configurations. Used to plot metric evolution.

---

This code supports reproduction of results and figures presented in the thesis:  
**“Study of properties of the Higgs boson at the ATLAS Experiment”**  
Erik Kvak, 2025

This project was developed in collaboration with my thesis supervisor  
**Mgr. Daniel Scheirich, Ph.D.**, and colleague **Patrik Ivan**.

GitHub: [https://github.com/kvakerik/Higgs_prediction_pT](https://github.com/kvakerik/Higgs_prediction_pT)
