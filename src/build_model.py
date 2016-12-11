from keras.models import Sequential
from keras.layers import Dense, Activation

def buildNNModel():
    # Define network strcuture
    model = Sequential()
    model.add(Dense(output_dim=128, input_dim=41))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=128, input_dim=128))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=5))
    model.add(Activation("softmax"))

    # compile model
    model.compile(loss="categorical_crossentropy",
                optimizer="adam", metrics=["accuracy"])
    return model
