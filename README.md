Tabular Deep Learning Package

This package is designed to facilitate multi-task deep learning with tabular data for genome-wide prediction across various real-world genomic datasets, including mice data, pig data, wheat data, 14-cancer microarray data, and a subset of Loblolly pine data.


Models Included: The package includes ten models for evaluation:LassoNet, LightGBM, TabR, TabNet, NODE, TabTransformer, FT-Transformer, AutoInt, Gandalf, SAINT. All models have been successfully implemented using PyTorch. Detailed information on the parameter spaces used for multi-trait genome-wide prediction (GWP) can be found in the accompanying Word document titled "Parameter Space for Different Tabular Deep Learning Models for GWP.docx."

Running the Models, take the LassoNet on the mice data as an example:

LassoNet in Colab Environment, running steps are as:

(1) Install Required Packages

Run the following command to install necessary dependencies:

pip install bayesian-optimization omegaconf

(2) Mount Your Google Drive Use the following commands:

from google.colab import drive

import os

drive.mount('/content/drive')

os.chdir('/content/drive/MyDrive/your-directory') 
(For me, it is: os.chdir('/content/drive/MyDrive/Colab Notebooks/lassonet_bo'))

Execute the Script Run the entry.py file to start the process.

Run the LassoNet using CPU Environment (on mice data):
(1) install necessary packages, e.g. pandas, numpy, bayes_opt, and so on.
(2) prepare the files.py, data.py, bo.py, data_mice.py. entrymice.py.
(3) run the model.



PyTorch Setup: Ensure that you have the appropriate computing environment set up for PyTorch, then run the files: LightGBM, NODE, and TabNet, these models can be executed directly within Anaconda or PyTorch environments.

For the other models, run the respective main files directly. 

*****Because the pig data is bigger than other four datasets and limited space, it is not possible for us to put all the datasets together. If you need, you can access this data use the link here: https://github.com/angelYHF/Pig-data.
