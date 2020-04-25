# Titanic Kaggle Project using Deep Learning
The project is Kaggle project titled "Titanic: Machine Learning from Disaster". We have investigated the project from Machine Learning (optimized random forest) Deep Learning (with tuned hyper-parameters) perspectives. The code available in this repository is related to Deep learning. We developed it using Python (v3.7) in PyCharm IDE. However, it also runs in Google colab notebook without having any problem. We explain here, how the code runs.

# Execution of the code is easy. Just follow the following steps:
1. Download the code (project_code_dl.py).
2. Download the train.csv and test.csv files (available in this repository).
3. Preserve the files in the root directory (of the code or Kaggle/Google-colab's notebook).
4. Execute the program. The code will dump a file Deep_Learning_Solution.csv in the root directory. This is the prediction to be tested on Kaggle.

# The code does the following things
1. It receives the input files (train.csv and test.csv).
2. Conducts data cleansing, missing data replacing and feature engineering on the both files combining them into one dataframe.
3. Applies (not tuned) deep learning (Multilayer Perceptrons) on the training dataset.
4. Then tunes hyper-parameters in the following order: epochs and mini-batch-size, gradient descent optimizer, architecture (hidden layers and number of neurons) and dropout probability.
5. At each stage of parameter tuning, it output the best parameters, best scores and the training-accuracy vs. epoch-number graph.
6. Once the tuning hyper-parameters is accomplished, it trains the network using training dataset and predicts the output on the test dataset.



