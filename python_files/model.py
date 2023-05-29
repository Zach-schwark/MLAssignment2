import tensorflow as tf
import numpy as np
import preprocessing as myData

(x_train, y_train) = myData.getTrainingData()
(x_validation, y_validation) = myData.getValidationData()
(x_test, y_test) = myData.getTestingData()
print(len(x_train[0]))


modelStructure = [71,50,30,10]
activationFunction = "relu"
numEpochs = 5
learningRate = 0.1
regularization = 0.1
optimizer = tf.keras.optimizers.SGD(learning_rate=learningRate)
loss_fn = tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM)

#setting up model
layers = [tf.keras.layers.InputLayer(modelStructure[0]),]
for i in range(1, len(modelStructure)):
        layers.append(tf.keras.layers.Dense(modelStructure[i], activation= activationFunction, bias_initializer= "ones",activity_regularizer=tf.keras.regularizers.L1(regularization)))
        
                
#initilizing model
model = tf.keras.models.Sequential(layers)

#gets predictions of untrained model/ first forward propagate
predictions = model(x_train[:1]).numpy()


#normalizing out put, to get ptedictions as probabilities
tf.nn.softmax(predictions).numpy()


#calculate last layers error/ final error
loss_fn(y_train[:1], predictions).numpy()


# "backpropogates" and compiles model with deltas annd gradeints etc....sets this stuffup for training
model.compile(optimizer= optimizer,
              loss=loss_fn,
              metrics=['accuracy'])


# trains the model on training data
history = model.fit(x_train, y_train, epochs=numEpochs)

# evaluates and validates model on validation data
model.evaluate(x_validation,  y_validation, verbose=2)

# evaluates on testing data
#model.evaluate(x_validation,  y_validation, verbose=2)










#things to consider changing to optimise

# model structure, layers nodes etc
# hyperparameters : learning rate, regularisation, number of epochs
# activation function
# loss function 
# how outputted

# change stuff using validating
# final test usingg testing 


# research into dropout