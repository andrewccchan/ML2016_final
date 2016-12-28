import numpy as np
import util
import csv
from document import Document

def predict(model):

    # Load test data. Modified from loadTrainData
    header = ["\"id\"", "\"tags\""]
    outFile = open("../submit.csv", "w")
    outFile.write(",".join(header) + "\n")

    with open("../data/test.csv", "r") as inFile:
        csvReader = csv.reader(inFile, quotechar='"', delimiter=',',
                        quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(csvReader, None) # skip header
        docs = [Document(l, "test") for l in csvReader]
        totalCount = util.calTotalCount(docs)
        for d in range(len(docs)):
            invVoc = {v: k for k, v in docs[d].vocab.iteritems()}
            docs[d].addTFIDF(totalCount, len(docs), True)
            fea = docs[d].getFeatures()
            fea, meanX, stdX = util.featureNorm(fea)
            # docs[d].debug()
            # labels = model.predict_classes(fea, batch_size=1, verbose=0)
            # print labels
            labels = model.predict_proba(fea, batch_size=1, verbose=0)
            # print labels
            posIdx = []
            posProb = []
            for ct1 in range(labels.shape[0]):
                if labels[ct1, 0] < labels[ct1, 1]:
                    posIdx.append(ct1)
                    posProb.append(labels[ct1, 1])
            sortIdx = np.argsort(posProb).tolist()
            realIdx = [posIdx[i] for i in sortIdx]
            tags = []

            realLen = len(realIdx)
            if realLen == 1:
                tags.append(invVoc[realIdx[-1]])
            elif realLen == 2:
                tags.append(invVoc[realIdx[-1]])
                tags.append(invVoc[realIdx[-2]])
            elif realLen > 2:
                tags.append(invVoc[realIdx[-1]])
                for ct1 in range(2):
                    tmp = -2 - ct1
                    if labels[realIdx[tmp]][1] > 0.8:
                        tags.append(invVoc[realIdx[tmp]])
            outFile.write("\"%d\"," % (docs[d].docId))
            outFile.write("\"" + " ".join(tags) + "\"\n")

    outFile.close()

def predict_const_number(model):

    # Load test data. Modified from loadTrainData
    header = ["\"id\"", "\"tags\""]
    outFile = open("../submit.csv", "w")
    outFile.write(",".join(header) + "\n")

    with open("../data/test.csv", "r") as inFile:
        csvReader = csv.reader(inFile, quotechar='"', delimiter=',',
                        quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(csvReader, None) # skip header
        docs = [Document(l, "test") for l in csvReader]
        totalCount = util.calTotalCount(docs)
        for d in range(len(docs)):
            invVoc = {v: k for k, v in docs[d].vocab.iteritems()}
            docs[d].addTFIDF(totalCount, len(docs), True)
            fea = docs[d].getFeatures()
            fea, meanX, stdX = util.featureNorm(fea)
            # docs[d].debug()
            # labels = model.predict_classes(fea, batch_size=1, verbose=0)
            # print labels
            labels = model.predict_proba(fea, batch_size=1, verbose=0)
            idx = np.argsort(labels[:,1]).tolist()

            tags = []
            for ct1 in range(3):
                tmp = idx[-1*ct1]
                tags.append(invVoc[tmp])

            outFile.write("\"%d\"," % (docs[d].docId))
            outFile.write("\"" + " ".join(tags) + "\"\n")

    outFile.close()

def predict_baseline():
    header = ["\"id\"", "\"tags\""]
    outFile = open("../submit.csv", "w")
    outFile.write(",".join(header) + "\n")

    with open("../data/test.csv", "r") as inFile:
        csvReader = csv.reader(inFile, quotechar='"', delimiter=',',
                        quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(csvReader, None) # skip header
        docs = [Document(l, "test") for l in csvReader]
        # totalCount = util.calTotalCount(docs)
        for d in range(len(docs)):
            # invVoc = {v: k for k, v in docs[d].vocab.iteritems()}
            # docs[d].addTFIDF(totalCount, len(docs), True)
            # fea = docs[d].getFeatures()
            tags = docs[d].getinTitle()

            outFile.write("\"%d\"," % (docs[d].docId))
            outFile.write("\"" + " ".join(tags) + "\"\n")

    outFile.close()
