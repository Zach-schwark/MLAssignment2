import tensorflow as tf
import numpy as np
import preprocessing as myData
from sklearn.decomposition import PCA

(x_train, y_train) = myData.getTrainingData()
(x_validation, y_validation) = myData.getValidationData()
(x_test, y_test) = myData.getTestingData()
f = open("evaluation.txt", "w")
f.write("")
f.close()
# second iteration 
pca = PCA(0.9)
pca.fit(x_train)
print(pca.n_components_)
x_train_pca = pca.transform(x_train)
x_validation_pca = pca.transform(x_validation)
print(np.shape(x_train_pca))
print(np.shape(x_train))


# research into dropout
#LR_arr = [0.1375, 0.14, 0.1425, 0.145]
#Reg_arr = [0.0625, 0.0637, 0.065, 0.0665]
#HLnodes_arr = [31]
#actFunc_arr = ["relu"]
#epochs_arr = [10]

LR_arr = [0.1375]
Reg_arr = [0.06325]
HLnodes_arr = [43]
actFunc_arr = ["relu"]
epochs_arr = [100]

# using list comprehension
# to compute all possible permutations
permutations = [[i, j, k,w,z] for i in HLnodes_arr
                for j in actFunc_arr
                for k in epochs_arr
                for w in LR_arr
                for z in Reg_arr]
for p in range(len(permutations)):
    modelStructure = permutations[p][0]
    activationFunction = permutations[p][1]
    numEpochs = permutations[p][2]
    learningRate =  permutations[p][3]
    regularization =  permutations[p][4]
    optimizer = tf.keras.optimizers.Adam(lr=learningRate)
    loss_fn = tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM)
    #setting up model
    layers = [tf.keras.layers.InputLayer(input_shape=(50,),ragged=False)]
    layers.append(tf.keras.layers.Dense(modelStructure, activation= activationFunction, bias_initializer= "ones",activity_regularizer=tf.keras.regularizers.L1(regularization)))
    layers.append(tf.keras.layers.Dense(10, activation= activationFunction, bias_initializer= "ones",activity_regularizer=tf.keras.regularizers.L1(regularization)))
    #initilizing model
    model = tf.keras.models.Sequential(layers)
    model.summary()
    print(model.layers[0])
    #gets predictions of untrained model/ first forward propagate
    predictions = model(x_train_pca[:1]).numpy()
    #normalizing out put, to get ptedictions as probabilities
    predictions = tf.nn.softmax(predictions).numpy()
    #predictions_final = np.zeros(len(predictions), dtype=int)
    #for i in range(len(predictions)):
    #        #print(len(np.where(predictions[i]==max(predictions[i]))))
    #        if(len(np.where(predictions[i]==max(predictions[i])))>=1):
    #                predictions_final[i]=int(np.where(predictions[i]==max(predictions[i]))[0][0])
    #        else:
    #                #print("hi")
    #                predictions_final[i]=int(np.where(predictions[i]==max(predictions[i]))[0])
    #
    #calculate last layers error/ final error
    loss_fn(y_train[:1], predictions).numpy()
    # "backpropogates" and compiles model with deltas annd gradeints etc....sets this stuffup for training
    model.compile(optimizer= optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                            mode="min", patience=50,
                                            restore_best_weights=True)
    print(model.weights[0])
    # trains the model on training data
    #history = model.fit(x_train, y_train, epochs=numEpochs)
    history = model.fit(x_train_pca, y_train, epochs=numEpochs,validation_data=(x_validation_pca,  y_validation),callbacks=[earlystopping])
    model.summary()
    print(model.weights[0])
    f = open("evaluation.txt", "a")
    f.write("\n \n Hyper Parameters:\n modelStructure ="+str(modelStructure)+"\n activationFunction = "+str(activationFunction)+"\n numEpochs = "+str(numEpochs) +"\n learningRate ="+str(learningRate)+"\n regularization = "+str(regularization )+"\n")
    f.write("loss="+str(model.evaluate(x_validation_pca,  y_validation, verbose=2)[0])+"\t accuracy= "+str(model.evaluate(x_validation_pca,  y_validation, verbose=2)[1])+ "\n")
    f.close()
    # evaluates and validates model on validation data
model.evaluate(x_validation_pca,  y_validation, verbose=2)


# evaluates on testing data
#model.evaluate(x_test,  y_test, verbose=2)
config = model.get_config()
model.save(filepath = "model.keras",save_format='tf')



# change stuff using validating
# final test usingg testing 



#predictions_test = model.predict_on_batch(x_test)
#predictions_test = tf.nn.softmax(predictions_test).numpy()
#predictions_final = np.zeros(len(predictions_test), dtype=int)
#for i in range(len(predictions_test)):
#        #print(np.where(predictions_test[i]==max(predictions_test[i])))
#        if(len(np.where(predictions_test[i]==max(predictions_test[i])))>1):
#                predictions_final[i]=int(np.where(predictions_test[i]==max(predictions_test[i]))[0][0])
#        else:
#                #print(np.where(predictions_test[i]==max(predictions_test[i]))[0])
#                predictions_final[i]=int(np.where(predictions_test[i]==max(predictions_test[i]))[0])