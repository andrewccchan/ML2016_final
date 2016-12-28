import eva
import keras

print "Predicting..."
model = keras.models.load_model("model.h5")
eva.predict_baseline()
