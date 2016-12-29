import keras
from sklearn.model_selection import train_test_split

def trainNN(model, X, y):
    X_train, X_test, y_train, y_test =  train_test_split(
        X, y, test_size=0.1, random_state=42)
    # earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                        # patience=1, verbose=0, mode='auto')
    weight = {0: 1, 1:50}
    # X_train = X_train[:,3]
    # X_test = X_test[:,3]
    model.fit(X_train, y_train,
            validation_data=(X_test, y_test),
            nb_epoch=7, batch_size=64,
            class_weight=weight)
            # callbacks=[earlyStopping])
    return model
