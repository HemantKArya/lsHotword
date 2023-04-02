
from __future__ import print_function
import numpy as np
from pydub import AudioSegment
import random
import sys
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment
print("lsHotwordTrainer")
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, Conv1D
from tensorflow.keras.layers import GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam




def Hmodel(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    with the help of Andrew ng and Hemant Kumar
    """
    
    X_input = Input(shape = input_shape)
    
 
    
    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X) # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    
    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                  # Batch normalization
    X = Dropout(0.8)(X)                                  # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)


    model = Model(inputs = X_input, outputs = X)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Dir For Dataset e.g. neg an pos')
    parser.add_argument('--inX', action='store', type=str, required=True)
    parser.add_argument('--inY', action='store', type=str, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--tx', action='store', type=int, default=5511)
    parser.add_argument('--nf', action='store', type=int, default=101)
    parser.add_argument('--ty', action='store', type=int, default=1375)
    args = parser.parse_args()
    Tx = args.tx # The number of time steps input to the model from the spectrogram
    n_freq = args.nf # Number of frequencies input to the model at each time step of the spectrogram
    Ty = args.ty # The number of time steps in the output of our model
    if os.path.isfile(args.inX):
        X = np.load(args.inX)
    else:
        print('Provide correct input for inX!')
    
    if os.path.isfile(args.inY):
        Y = np.load(args.inY)
    else:
        print('Provide correct input for inX!')
    print(X.shape)
    print(Y.shape)
    assert X.ndim == 3, "Error: X not have correct dimentions"
    assert Y.ndim == 3, "Error: Y not have correct dimentions"
    assert Y.shape[1] == Ty, "Error: Y not have correct dimentions"
    assert X.shape[1] == Tx, "Error: X not have correct dimentions"
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=37)
    model = Hmodel(input_shape = (Tx, n_freq))
    model.summary()
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    epoch = args.epochs
    model.fit(X,Y,batch_size=10,epochs=epoch)
    while True:
        print("Wanna train more!! (y/n)")
        ch = input('>> ')
        if ch == 'y':
            model.fit(X,Y,batch_size=5,epochs=5)
        else:
            break
    model.save("model.h5")
    print("Model Saved !")

 

