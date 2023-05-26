import numpy as np
import tensorflow as tf
import math

f = open("data/traindata.txt", "r")
#print(f.read())
xstr = f.read()
x = xstr.splitlines()
x_vals_list = []
for i in range(math.floor(len(x)/1000)):
    xArr = x[i].split(",")
    tempArr = np.zeros(len(xArr))
    for j in range(len(xArr)):
        tempArr[j]= float(xArr[j])
    x_vals_list.append(np.array(tempArr))
x_vals=np.array(x_vals_list)

f = open("data/trainlabels.txt", "r")
ystr = f.read()
y = ystr.splitlines()
y_vals_list = []
for i in range(math.floor(len(x)/1000)):
    yArr = y[i].split(",")
    tempArr = np.zeros(len(yArr))
    y_vals_list.append(int(yArr[0]))
y_vals=np.array(y_vals_list)

#70% training, 20% validation, 10% testing
x_trainList = []
y_trainList = []
#training data
for i in range(math.floor(0.70*len(x_vals))):
    x_trainList.append(x_vals[i])
x_train=np.array(x_trainList)
for i in range(math.floor(0.70*len(y_vals))):
    y_trainList.append(y_vals[i])
y_train=np.array(y_trainList)

x_validationList = []
y_validationList = []
#validation data
for i in range(math.floor(0.70*len(x_vals)),math.floor(0.90*len(x_vals))):
    x_validationList.append(x_vals[i])
x_validation=np.array(x_validationList)
for i in range(math.floor(0.70*len(y_vals)),math.floor(0.90*len(y_vals))):
    y_validationList.append(y_vals[i])
y_validation=np.array(y_validationList)

x_testingList = []
y_testingList = []
#testing data
for i in range(math.floor(0.90*len(x_vals)),len(x_vals)):
    x_testingList.append(x_vals[i])
x_testing=np.array(x_testingList)
for i in range(math.floor(0.90*len(y_vals)),len(y_vals)):
    y_testingList.append(y_vals[i])
y_testing=np.array(y_testingList)

def getTrainingData():
    return x_train, y_train

def getValidationData():
    return x_validation, y_validation

def getTestingData():
    return x_testing, y_testing