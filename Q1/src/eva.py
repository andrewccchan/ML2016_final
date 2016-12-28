import numpy as np
import pickle
import util

def predict(model):
    # Load data
    with open("../obj/codes.p", "r") as c:
        codes = pickle.load(c)
    # print codes
    print("Reading test data")
    with open("../data/test.in", "r") as raw:
        testData = [util.processRaw(l, codes) for l in raw]

    classes = model.predict_classes(testData, batch_size=32)

    header = ["id", "label"]
    with open("../submit.csv", "w") as sub:
        sub.write(",".join(header) + "\n")
        for ct1 in range(1, classes.shape[0]+1):
            sub.write("%d,%d" % (ct1, classes[ct1-1]) + "\n")
