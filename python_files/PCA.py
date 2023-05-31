import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import zscore # used for standardization
from sklearn.preprocessing import StandardScaler
import preprocessing as myData


(x_train, y_train) = myData.getTrainingData()
(x_validation, y_validation) = myData.getValidationData()
(x_test, y_test) = myData.getTestingData()
#print(x_train)


x = StandardScaler().fit_transform(x_train) # normalizing the features


def normalize(data):
 ## creates a copy of data
 #X = tf.identity(data)
 ## calculates the mean
 #X -=tf.reduce_mean(data, axis=0)
    x_features = np.transpose(data)
    for xfeature in x_features:
        mean=np.mean(xfeature)
        std=np.std(xfeature)
        for z in xfeature:
            z=(z-mean)/std
    #normalized =  np.transpose(x_features)
    return x_features

normalized_features = normalize(x_train)
#print(np.shape(normalized_features))
covariance_matrix = np.cov(normalized_features)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = list(map(eigenvalues.__getitem__, indices))
sorted_eigenvectors = list(map(eigenvectors.T.__getitem__, indices))
#print(np.shape(sorted_eigenvectors[0]))
principal_components = np.dot(np.transpose(normalized_features) ,sorted_eigenvectors[0])
variance_explained = [i/sum(sorted_eigenvalues) for i in sorted_eigenvalues]

#print(np.shape(principal_components))
#print(np.shape(normalized_features))
print(variance_explained)


