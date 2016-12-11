from __future__ import print_function
import numpy as np
from os.path import isfile
import util
import build_model
import eva
import train

if not isfile("../obj/rawData.npy"):
    data = util.getTrainData()
    np.save("../obj/rawData.npy", data)
else:
    data = np.load("../obj/rawData.npy")

model = build_model.buildNNModel()

print(data[:,-1].shape)
print("\n".join(map(str, data[:,-1].tolist())))
print("Training NN")
train.trainNN(model, data)

print("Predicting test data")
eva.predict(model)
