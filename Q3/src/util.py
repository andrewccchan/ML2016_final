import re
from document import Document
import csv
import numpy as np

def loadPredictions():
    pre = dict()
    with open("../submit.csv", "r") as preFile:
        csvReader = csv.reader(preFile, quotechar='"', delimiter=',',
                        quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(csvReader, None) # skip header
        for l in csvReader:
            pre[int(l[0])] = l[1].split(" ")

    return pre

def loadTestDataWOFeatures():
    with open("../data/test.csv", "r") as inFile:
        print "Load test data"
        csvReader = csv.reader(inFile, quotechar='"', delimiter=',',
                        quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(csvReader, None) # skip header
        docs = [Document(l, "test") for l in csvReader]

    return docs

    return docs
def loadTrainData(inputCorps):
    X = []
    y = []
    for cor in inputCorps:
        with open("../data/"+cor+".csv", "r") as corFile:
            print "#############################"
            print "Processing data %s" % (cor)
            csvReader = csv.reader(corFile, quotechar='"', delimiter=',',
                            quoting=csv.QUOTE_ALL, skipinitialspace=True)
            next(csvReader, None) # skip header
            docs = [Document(l, cor) for l in csvReader]
            print "Collecting totalCount"
            totalCount = calTotalCount(docs)
            print "Generating features"
            for d in range(len(docs)):
                docs[d].addTFIDF(totalCount, len(docs), True)
                X.append(docs[d].getFeatures())
                y.append(docs[d].getLabels())
            print "#############################"
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return (X, y)

def loadStopWords():
    print "loading stop words"
    with open("../data/stopwords.txt", "r") as f:
        stopW = {l.strip():1 for l in f}
    return stopW

stopWords = loadStopWords()
def cleanInput(raw):
    htmlCleaner = re.compile('<.*?>')
    rawClean = re.sub(htmlCleaner, "", raw)
    word_split = re.compile('[^a-zA-Z0-9_\\+/]')
    # TODO: Try not to split "-"
    ret = []
    for word in word_split.split(rawClean):
        word = word.strip().lower()
        if word not in stopWords and word != "":
            ret.append(word)
    return ret

def procInput(raw):
    ret = []
    for r in raw:
        ret.append(cleanInput(r))
    ret[0] = int(ret[0][0])
    return ret

def calTotalCount(docs):
    totalCount = dict()
    for d in docs:
        for key in d.vocab:
            if key in totalCount:
                totalCount[key] += 1
            else:
                totalCount[key] = 1
    return totalCount

def featureNorm(X, meanX=None, stdX=None):
    if meanX == None:
        meanX = np.mean(X, axis=0)
    if stdX == None:
        stdX = np.std(X, axis=0)
    return np.divide((X - meanX), stdX), meanX, stdX
