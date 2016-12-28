import util
import math
import numpy as np

class Document:

    meaning_less = ['p','would','could','via','emp','two','must','make',
                        'e','c','using','r','vs','versa','based','three']

    def __init__(self, raw, cor):
        self.corName = cor
        if self.corName == "test":
            self.docId, self.title, self.cont = util.procInput(raw)
        else:
            self.docId, self.title, self.cont, self.tag = util.procInput(raw)
        self.vocab = dict()
        self.wid = 0
        self.feaArr = []
        self.genFeatures()


    def genFeatures(self):
        # Combine title, contents and tag
        fDoc = self.__getFullDoc()

        # Collect tf
        tf = dict()
        wordCnt = 0
        for w in fDoc:
            if w not in self.vocab:
                self.vocab[w] = self.wid
                self.wid += 1
                tf[w] = 1
            else:
                tf[w] += 1
            wordCnt += 1

        tmpTF = [None]*len(tf)
        for key, val in tf.iteritems():
            tmpTF[self.vocab[key]] = float(val) / wordCnt
        self.feaArr.append(tmpTF)

        # Collect intitle feature
        assert(len(tf) == len(self.vocab))
        inTit = [0] * self.wid
        for w in self.title:
            try:
                inTit[self.vocab[w]] = 1
            except KeyError:
                pass
        self.feaArr.append(inTit)

    # totalCount: Dict with key = word, val = count in corpus
    def addTFIDF(self, totalCount, D, add_idf = True):
        tmpIdf = [0] * self.wid
        for word, idx in self.vocab.iteritems():
            tmpIdf[idx] = math.log10(float(D) / totalCount[word])
        # Add IDF
        if add_idf:
            self.feaArr.append(tmpIdf)
        # Calculate TF-IDF
        self.feaArr.append([a*b for a, b in zip(self.feaArr[0], tmpIdf)])

    def getLabels(self):
        assert(self.tag != "")
        ret = [0] * self.wid
        for word, idx in self.vocab.iteritems():
            ret[idx] = 1 if (word in self.tag) else 0
        return np.asarray(ret)

    def getFeatures(self):
        return np.asarray(self.feaArr).T

    # Verbosely print debug messages
    def debug(self, itemNum=7):
        invVoc = {v: k for k, v in self.vocab.iteritems()}
        for ct1 in range(itemNum):
            print "%s: %f, %f, %f, %f" %\
             (invVoc[ct1], self.feaArr[0][ct1],
             self.feaArr[1][ct1], self.feaArr[2][ct1],
             self.feaArr[3][ct1])

    # Return words which appear in the title
    def getinTitle(self):
        com = []
        common = set(self.cont).intersection(self.title)
        if len(common) ==0:
            for t in self.title:
                if t not in self.meaning_less:
                    com.append(t)
            return com
        else:
            return common
    # Combine title, contents
    def __getFullDoc(self):
        return self.title + self.cont
