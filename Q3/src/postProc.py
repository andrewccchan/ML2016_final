import util
import collections

# Load test data
docs = util.loadTestDataWOFeatures()

# Load predictions
pres = util.loadPredictions()

# Merge all documents
fullDoc = []
for d in docs:
    fullDoc.extend(d.getFullDoc())

# later and previous word occurrence frequency matrix
postDict = dict()
prevDict = dict()

# Collect all tags
allTags = dict()
for p in pres:
    for t in pres[p]:
        allTags[t] = 1

# Fill postDict and prevDict
print "Generating prev/post dict"
for idx in range(1, len(fullDoc)-1):
    w = fullDoc[idx]
    if w in allTags:
        if w not in postDict:
            postDict[w] = dict()
            prevDict[w] = dict()

        later = fullDoc[idx+1]
        if later in postDict[w]:
            postDict[w][later] += 1
        else :
            postDict[w][later] = 0

        prev = fullDoc[idx-1]
        if prev in prevDict[w]:
            prevDict[w][prev] += 1
        else :
            prevDict[w][prev] = 0

# Compute the most probable next/prev word dict
print "Generating max prob. dict"
maxPostDict = dict()
maxPrevDict = dict()
for key, val in postDict.iteritems():
    maxKey = ""
    maxVal = -1
    for a, b in val.iteritems():
        if b > maxVal:
            maxKey = a
            maxVal = b
    rec = [maxKey, maxVal]
    maxPostDict[key] = rec

for key, val in prevDict.iteritems():
    maxKey = ""
    maxVal = -1
    for a, b in val.iteritems():
        if b > maxVal:
            maxKey = a
            maxVal = b
    rec = [maxKey, maxVal]
    maxPrevDict[key] = rec
# print maxPostDict
print "Generating final tags"
output = dict()
cntThresh = 500
# Predict phrases for each tag
for idx, tags in pres.iteritems():
    finalTags = []
    for t in tags:
        try:
            postCand = maxPostDict[t]
            prevCand = maxPrevDict[t]
        except KeyError:
            continue

        if postCand[1] > prevCand[1] and postCand[1] > cntThresh:
            finalTags.append(t+"-"+postCand[0])
        elif postCand[1] < prevCand[1] and prevCand[1] > cntThresh:
            finalTags.append(prevCand[0]+"-"+t)
        else:
            finalTags.append(t)
    output[idx] = finalTags

# Sort output by index
print "Writing results to file"
output = collections.OrderedDict(sorted(output.items()))

header = ["\"id\"", "\"tags\""]
outFile = open("../submit_extend.csv", "w")
outFile.write(",".join(header) + "\n")

for idx, tags in output.iteritems():
    sTags = set(tags)
    outFile.write("\"%d\"," % (idx))
    outFile.write("\"" + " ".join(sTags) + "\"\n")
