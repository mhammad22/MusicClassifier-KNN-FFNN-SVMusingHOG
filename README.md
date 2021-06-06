# MusicClassifier-KNN-FFNN-SVMusingHOG
 
# Installtion Instruction:
For this Project, you need compiler for Java to run this file. There are many compilers like Pycharm, Vscode and many others which support java.
* Pycharm: https://www.jetbrains.com/pycharm/download/
* VsCode: https://code.visualstudio.com/download

# Compilation Instruction:
Just Install the compiler from their official website and Run project. In case you are using Vscode then you have to download the dependency for Python file.

# Working:
The repositiory contains different folders for different training scripts. Each of the trained models performed differently and has different success rate. These include:

1. kNN (k Neareast Neighbors)
2. FFNN (Feed Forward Neural Network)
3. SVM using HOG (Support Vector Machines using Histogram of Oriented Gradients features)
The dataset for the models is MNIST dataset which is a huge dataset containing a handsome amount of records. The images of characters are converted to pixels with their intensity ranging from 0-255.

A report is also attached in which a detailed analysis of the models is provided alongwith the accuracies of each model and their configurations.

# How to Run ?
Download the MNIST Handwriting Recognition dataset. It's easily available on Kaggle. Donwload the train.py files for each model and change the dataset paths respectively. Run the model and it will provide you with training and test accuracies.
