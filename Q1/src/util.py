import pickle
import numpy as np

# Ouput: training data after conversion
def getTrainData():
    with open("../obj/codes.p", "r") as c:
        codes = pickle.load(c)

    # Convert codes of detailed attack types to codes of 4 attack types
    with open("../data/training_attack_types.txt") as tat:
        typeCode = { "normal": 0, "dos":1, "u2r":2, "r2l":3, "probe":4}
        typeMap = dict()
        for row in tat:
            fields = row.strip("\n").split(" ")
            typeMap[fields[0]] = typeCode[fields[1]]
        typeMap["normal"] = 0
        for key in codes[-1]:
            codes[-1][key] = typeMap[key]

    print("Reading data")
    with open("../data/train", "r") as raw:
        rawData = [processRaw(l, codes) for l in raw]
    return np.asarray(rawData)

# Input: raw=a single row of raw data,
#        codes=a list of code dictionary
# Output: Processed raw data
def processRaw(raw, codes):
    raw = raw.rstrip(".\n").split(",")
    ret = raw
    for idx, field in enumerate(raw):
        try:
            ret[idx] = float(field)
        except ValueError:
            # print idx
            try:
                ret[idx] = codes[idx][field]
            except KeyError:
                ret[idx] = 0
    return ret

# Input: training data in lists of list
# Output string codes
def analyzeRawData(rawData):
    feaLen = len(rawData[0])
    codeList = [dict() for tmp in range(feaLen)]
    dictIdx = [0] * feaLen

    for row in rawData:
        for idx, field in enumerate(row):
            try:
                float(field)
            except ValueError:
                if field not in codeList[idx]:
                    codeList[idx][field] = dictIdx[idx]
                    dictIdx[idx] += 1
    # write codesList to pickle
    with open("../obj/codes.p", "wb") as f:
        pickle.dump(codeList, f, pickle.HIGHEST_PROTOCOL)
    print codeList
