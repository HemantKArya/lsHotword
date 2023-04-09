
from numpy import load as npload
from os import path
print("lsHotwordTrainer")
import argparse
from sklearn.model_selection import train_test_split
try:
    from tensorflow.keras.optimizers.legacy import Adam
except:
    from tensorflow.keras.optimizers import Adam
from . import hotword as ht




def main():
    parser = argparse.ArgumentParser(description='Dir For Dataset e.g. neg an pos')
    parser.add_argument('--inX', action='store', type=str, required=True)
    parser.add_argument('--inY', action='store', type=str, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--tx', action='store', type=int, default=5511)
    parser.add_argument('--nf', action='store', type=int, default=101)
    parser.add_argument('--ty', action='store', type=int, default=1375)
    parser.add_argument('--bsize', action='store', type=int, default=10)
    args = parser.parse_args()
    Tx = args.tx # The number of time steps input to the model from the spectrogram
    n_freq = args.nf # Number of frequencies input to the model at each time step of the spectrogram
    Ty = args.ty # The number of time steps in the output of our model
    if path.isfile(args.inX):
        X = npload(args.inX)
    else:
        print('Provide correct input for inX!')
    
    if path.isfile(args.inY):
        Y = npload(args.inY)
    else:
        print('Provide correct input for inX!')
    print(X.shape)
    print(Y.shape)
    assert X.ndim == 3, "Error: X not have correct dimentions"
    assert Y.ndim == 3, "Error: Y not have correct dimentions"
    assert Y.shape[1] == Ty, "Error: Y not have correct dimentions"
    assert X.shape[1] == Tx, "Error: X not have correct dimentions"
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=37)
    model = ht.Hmodel(input_shape = (Tx, n_freq))
    model.summary()
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    epoch = args.epochs
    model.fit(X,Y,batch_size=args.bsize,epochs=epoch)
    model.save("model.h5")
    print("Model Saved !")

 

