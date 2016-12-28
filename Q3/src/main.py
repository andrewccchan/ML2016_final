import util
from build_model import buildNNModel
from keras.utils.np_utils import to_categorical
import train
import eva
import keras

inputCorps = ['cooking', 'crypto', 'diy', 'robotics', 'travel']
(X, y) = util.loadTrainData(inputCorps)
X, meanX, stdX = util.featureNorm(X)
y = to_categorical(y)

print "Training data size: %d" % (X.shape[0])
assert(X.shape[0] == y.shape[0])

print "Training..."
model = buildNNModel()

train.trainNN(model, X, y)

model.save("model.h5")

# print "Predicting..."
# modelLoad = keras.models.load_model("model.h5")
# eva.predict(modelLoad, meanX, stdX)
