import tensorflow as tf
import numpy as np
import preprocessing as myData

(x_train, y_train) = myData.getTrainingData()
(x_validation, y_validation) = myData.getValidationData()
(x_test, y_test) = myData.getTestingData()
print(len(x_train[0]))

#model structure
modelStructure = [71,50,30,10]
activationFunction = "relu"

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(modelStructure[0]),
    tf.keras.layers.Dense(modelStructure[1], activation= activationFunction),
    tf.keras.layers.Dense(modelStructure[2], activation= activationFunction),
    tf.keras.layers.Dense(modelStructure[3], activation= activationFunction)
])

predictions = model(x_train[:1]).numpy()
predictions
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=50)
model.evaluate(x_validation,  y_validation, verbose=2)

#things to consider changing to optimise

# model structure, layers nodes etc
# hyperparameters : learning rate, regularisation, number of epochs
# activation function
# loss function 
# how outputted