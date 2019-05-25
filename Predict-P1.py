# Import modules
import numpy as np
from keras.models import Sequential

from keras.layers import Dense,SimpleRNN

from time import time

from keras.callbacks import TensorBoard

import argparse

import subprocess

import os

### set seed for training
np.random.seed(1234)

#### Execute Rscript to generate synthetic data
subprocess.call("Rscript Synth_Lunch.R",shell=True)


### Command line arugments
parser = argparse.ArgumentParser()

parser.add_argument('--days',type=int,help="Number of days to make predictions on glucose",default=1)

parser.add_argument('--A1C',type=bool,help="Would you like to calculate A1C?",default=False)

args = parser.parse_args()

days = args.days

A1C = args.A1C



### load data files

p1_current = np.genfromtxt('Data/p1-current_synth.csv',delimiter=',')

p1_next = np.genfromtxt('Data/p1-next_synth.csv',delimiter=",")


split = .3

steps = 3





from sklearn.model_selection._split import train_test_split

import keras.backend as K


from keras import optimizers







current = p1_current





current = current[1:len(current)]


MAX = current.max()

MIN = current.min()


next = p1_next


next = next[1:len(next)]



if(next.max() > current.max()):
    MAX = next.max()

if(next.min() < current.min()):
    MIN = next.min()



### Scale values
X = (current - MIN) / (MAX - MIN)

X = X.reshape(len(X),1,steps)

Y = (next - MIN) / (MAX - MIN)

Y = Y.reshape(len(Y),1,steps)




#random.seed(1)
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=split)




def Gavg(y_true,y_pred):
    ## This function computes error in how many glucose points on average we deviate

    error = Eavg(y_true,y_pred)


    gval = error * (MAX - MIN)

   # L1 = K.sum(gval)

    return gval








def percent_error(y_true,y_pred):
    error = Eavg(y_true,y_pred)




    return K.abs(K.mean((error/y_true) * 100))

def Eavg(y_true,y_pred):
  return  K.mean(K.abs(y_true - y_pred))





model = Sequential()





model.add(SimpleRNN(steps,return_sequences=True,input_shape=(1,steps),activation='tanh'))

model.add(SimpleRNN(steps,return_sequences=True))
model.add(SimpleRNN(steps,return_sequences=True))

model.add(SimpleRNN(steps,return_sequences=True,activation='elu'))




model.compile(optimizer=optimizers.sgd(lr=.065,momentum=.95,decay=.2),loss= 'mean_absolute_error')


#model.summary()




tensorboard = TensorBoard(log_dir="logs/{}".format(time()),histogram_freq=0)


start = time()
model.fit(X_train,y_train,epochs =  10,batch_size= 15,verbose=0)

end = time()
model.save("Models/P1-D2D.HDF5")



print( end - start)


#os.system('clear')

readings = []






######### Welcoming message

print("Sequential Glucose Predictor ")


print()



lunch = int(input("Enter your lunch reading "))

readings.append(lunch)

dinner = int(input("Enter your dinner reading "))

readings.append(dinner)

bed = int(input("Enter your bed reading "))

readings.append(bed)


print(readings)


readings = np.array(readings)


#### Function to compute A1C
def calc_A1C(predictions):
    full_sum = 0
    for x in range(len(predictions)):
        full_sum += sum(predictions[x][0][0][:3])



    average_glucose = full_sum / (days * steps)

    A1C_val = (46.7 + average_glucose) / 28.7

    return A1C_val




readings =  (readings - MIN) / (MAX - MIN)

print(readings)

readings = readings.reshape(1,1,3)

if A1C:
    days = 90

predictions = []



for x in range(days):


    if A1C == True:
        preds = model.predict(readings, verbose=0)
    else:
        preds = model.predict(readings, verbose=1)


    glucose_val = preds * (MAX - MIN) + MIN
    if A1C == False:

        print("Day",x,glucose_val,":")

    predictions.append(glucose_val)

    readings = preds




if A1C:
    print("Your Pedicted A1C is  ",round(calc_A1C(predictions),2),"%")




