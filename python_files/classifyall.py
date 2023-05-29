import numpy as np

# need to:
# import model/ get trained model to use
# read in data from file testdata.txt
# predicts data lables
# writes predicted lables top file testlables.txt

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

x_test = getData()

model.predict(x_test)

#write output to textfile 