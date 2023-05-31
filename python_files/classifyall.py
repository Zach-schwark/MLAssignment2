import numpy as np
import tensorflow as tf
import preprocessing as myData

# need to:
# import model/ get trained model to use
# read in data from file testdata.txt
# predicts data lables
# writes predicted lables top file testlables.txt

# load model 
model = tf.keras.models.load_model('model.h5')

# check model info 
#model.summary()

(x_test, y_test) = myData.getTestingData()

def getData():
    f = open("data/testdata.txt", "r")
    #print(f.read())
    xstr = f.read()
    x = xstr.splitlines()
    x_vals_list = []
    for i in range(len(x)):
        xArr = x[i].split(",")
        tempArr = np.zeros(len(xArr))
        for j in range(len(xArr)):
            tempArr[j]= float(xArr[j])
        x_vals_list.append(np.array(tempArr))
    x_vals=np.array(x_vals_list)
    
    return x_vals

#x_test = getData()

predictions = model.predict_on_batch(x_test)
predictions = tf.nn.softmax(predictions).numpy()
predictions_final = np.zeros(len(predictions), dtype=int)
for i in range(len(predictions)):
        #print(len(np.where(predictions[i]==max(predictions[i]))))
        if(len(np.where(predictions[i]==max(predictions[i])))>=1):
                predictions_final[i]=int(np.where(predictions[i]==max(predictions[i]))[0][0])
        else:
                #print("hi")
                predictions_final[i]=int(np.where(predictions[i]==max(predictions[i]))[0])
#print(str(predictions_final))
#print(len(predictions[0]))
f = open("testlabels.txt", "w")
for i in range(len(predictions_final)):
    f.write(str(predictions_final[i])+"\n")
f.close()
