import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hpsklearn import HyperoptEstimator, any_classifier, random_forest, svc, knn
import hyperopt.pyll.stochastic
from datetime import datetime


#Reading data from CSV File
data = pd.read_csv("train.csv").to_numpy()
test = pd.read_csv("test.csv").to_numpy()

test_rows, test_cols=test.shape
print("Test Rows = ", test_rows, "\nTest Cols = ", test_cols)

rows,cols = data.shape
print("----------------------------\nData Rows = ", rows, "\nData Cols = ", cols)

trainRows = int(0.15*rows)

#Setting training data
x = data[0:21000,1:]
y = data[0:21000,0]

#Setting test data
xtest = data[21000:,1:]
actual_label = data[21000:,0]

# # 10 fold cross validation
# # Creating odd list K for KNN
# neighbors_range = list(range(1,15))
# # empty list that will hold cv scores
# cv_scores = [ ]
# #perform 10-fold cross-validation
# for K in neighbors_range:
#     print("Iteration ", K)
#     knn = neighbors.KNeighborsClassifier(n_neighbors = K, n_jobs = -1)
#     scores = cross_val_score(knn, x, y, cv = 5)
#     print(scores)
#     cv_scores.append(scores.mean())

# # Changing to mis classification error
# mse = [1-x for x in cv_scores]
# print(mse)
# # determing best k
# optimal_k = neighbors_range[mse.index(min(mse))]
# print("The optimal no. of neighbors is {}".format(optimal_k))

# pd.DataFrame({"K":[i for i in range(1,15,2)], "Accuracy":cv_scores}).set_index("K").plot.bar(figsize= (9,6),ylim=(0.78,1.00),rot=0)
# pt.show()

# exit()

knn = neighbors.KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)    
knn.fit(x, y)

#Testing the data
# d = xtest[3000]
# d.shape = (28,28)
# pt.imshow(255-d,cmap='gray')
# print("\n\nActual Value: ", actual_label[3000])
# pt.show()

print("Starting Pediction")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Starting Time =", current_time)

#Running the model on test data
p = knn.predict(xtest)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Ending Time =", current_time)
print("Done Predicting")

# Finding out the accuracy of trained model
count = 0
for i in range (0,21000):
    count += 1 if p[i]==actual_label[i] else 0

print("Accuracy of kNN Model: ", (count/(21000))*100)

# submit = pd.DataFrame(p,columns=["Label"])
# submit["ImageId"] = pd.Series(range(1,(len(p)+1)))
# submission = submit[["ImageId","Label"]]
# print(submission.shape)
# submission.to_csv("submission.csv",index=False)

########## Extras ##########
# params = {'n_neighbors':[3,4,5,6,7,8,9,10],
#           'leaf_size':[1,2,3,5],
#           'weights':['uniform', 'distance'],
#           'algorithm':['auto', 'ball_tree','kd_tree','brute'],
#           'n_jobs':[-1]}
# #Making models with hyper parameters sets
# model1 = GridSearchCV(knn, param_grid=params, n_jobs=-1)
# #Learning
# model1.fit(x,y)
# #The best hyper parameters set
# print("Best Hyper Parameters:\n",model1.best_params_)
# exit()