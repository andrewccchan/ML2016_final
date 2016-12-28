from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

def buildNNModel():
    # Define network strcuture
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=4))
    model.add(Activation("relu"))
    # model.add(Dense(output_dim=64, input_dim=128))
    # model.add(Activation("relu"))
    model.add(Dense(output_dim=2))
    model.add(Activation("softmax"))

    # compile model
    # opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999,
    #             epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy",
                optimizer="adam", metrics=["accuracy"])
    return model
