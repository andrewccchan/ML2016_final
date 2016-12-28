import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import keras

# QUESTION: Need fea. normalization in NN?
def trainNN(model, data):
    X = data[:,:-1]
    y = data[:,-1]
    y = to_categorical(y)

    X_train, X_test, y_train, y_test =  train_test_split(
        X, y, test_size=0.1, random_state=42)

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                        patience=0, verbose=0, mode='auto')
    model.fit(X_train, y_train,
            validation_data=(X_test, y_test),
            nb_epoch=5, batch_size=32,
            callbacks=[earlyStopping])
    return model
