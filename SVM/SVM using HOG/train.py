
# import the necessary packages
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from utils.hog import HOG
import numpy as np
import pandas as pd
import argparse

# load the dataset and initialize the data matrix
data = np.genfromtxt(r"C:\Users\Danish Ali\Python\train.csv", delimiter = ",", dtype = "uint8")
target = data[1:, 0]
digits = data[1:, 1:].reshape(data.shape[0] - 1, 28, 28) # remove head line

print("Train Data Shape: ", data.shape)

data = []

# initialize the HOG descriptor with the best score from evaluation
# for 0.9707
hog = HOG(orientations = 6, pixelsPerCell = (4, 4),
	cellsPerBlock = (4, 4), block_norm = 'L2-Hys')

# loop over the images
for image in digits:
	hist = hog.describe(image)
	data.append(hist)

# train the model
model = LinearSVC(random_state = 42)
# model = RandomForestClassifier(n_estimators = 50) # train RTC model
print("Tarining model on the given file...")
model.fit(data, target)
print("Model trained on given file...")

# dump the model to file
# joblib.dump(model, args["model"])

data = np.genfromtxt(r"C:\Users\Danish Ali\Python\test.csv", delimiter = ",", dtype = "uint8")
digits_test = data[1:, :].reshape(data.shape[0] - 1, 28, 28) # remove head line

print("Test data shape", digits_test.shape)

data = []

# Prepare images and calculates features (HOG)
for image in digits_test:
	hist = hog.describe(image)			
	data.append(hist)

# Create predictions
print("Predicitng values for data...")
predicted = model.predict(data)
print("Values for data predicted...")

print("Exporting data to .csv file")
submit = pd.DataFrame(predicted, columns=["Label"])
submit["ImageId"] = pd.Series(range(1,(len(predicted)+1)))
submission = submit[["ImageId","Label"]]
print(submission.shape)
submission.to_csv("submission.csv",index=False)
