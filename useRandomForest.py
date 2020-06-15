from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext
import math
import json
import numpy as np
import time
sc = SparkContext('local[*]', 'ModelCF')
sc.setLogLevel("ERROR")

start = time.time()

inputTrainFile = sys.argv[1]
inputUserFile = sys.argv[2]
inputTipFile = sys.argv[3]
inputCheckInFile = sys.argv[4]
inputBusinessFile = sys.argv[5]
inputTestFile = sys.argv[6]
outputFile = sys.argv[7]
########################################################################################
#Input file RDD

trainData = sc.textFile(inputTrainFile)
trainData = trainData.map(lambda row: row.split(',')).filter(lambda x: x[0] != "user_id").persist()
trainDataRdd = trainData.map(lambda x:(x[0],x[1]))
testData = sc.textFile(inputTestFile)
testData = testData.map(lambda row: row.split(',')).filter(lambda x: x[0] != "user_id").persist()
testEachDataRdd = testData.map(lambda x: (x[0], x[1]))

#########################################################################################
# Addidional file Rdd
userRdd = sc.textFile(inputUserFile).map(json.loads)
userAttributesRdd = userRdd.map(lambda x:(x['user_id'],(x['review_count'],x['average_stars']))).persist()
userAttributesRddDict = userRdd.map(lambda x:(x['user_id'],(x['review_count'],x['average_stars']))).collectAsMap()
checkInRdd = sc.textFile(inputCheckInFile).map(json.loads)
checkAttributesRdd = checkInRdd.map(lambda x:(x['business_id'],x['time']))
checkAttributesDict = checkAttributesRdd.map(lambda x:(x[0],sum(x[1].values()))).collectAsMap()
businessRdd = sc.textFile(inputBusinessFile).map(json.loads)
businessAttrDict = businessRdd.map(lambda x:(x['business_id'],(x['stars'],x['review_count']))).collectAsMap()

checkIn = float(sum(checkAttributesDict.values())/len(checkAttributesDict.values()))
userReviewCount = userRdd.map(lambda x:(x['review_count'])).collect()
userAvgStars = userRdd.map(lambda x:(x['average_stars'])).mean()
busReviewCount = businessRdd.map(lambda x:(x['review_count'])).mean()
busStars = businessRdd.map(lambda x:(x['stars'])).mean()

##########################################################################################
#Converting ids to integers
usersDictionary = {}
businessDictionary = {}

userIds = trainData.map(lambda x:x[0]).distinct().collect()
businessIds = trainData.map(lambda x:x[1]).distinct().collect()

testUserIds = testData.map(lambda x:x[0]).distinct().collect()
testBusinessIds = testData.map(lambda x:x[1]).distinct().collect()
counter_i=0
for counter_i in range(len(userIds)):
    usersDictionary[userIds[counter_i]] = counter_i

for j in range(len(testUserIds)):
    if testUserIds[j] not in usersDictionary:
        usersDictionary[testUserIds[j]] = counter_i+1
        counter_i = counter_i+1

counter_i=0
for counter_i in range(len(businessIds)):
    businessDictionary[businessIds[counter_i]] = counter_i

for j in range(len(testBusinessIds)):
    if testBusinessIds[j] not in businessDictionary:
        businessDictionary[testBusinessIds[j]] = counter_i+1
        counter_i = counter_i+1

#################################################################################
def featureCombine(row):
    global userReviewCount,userAvgStars,busReviewCount,busStars,checkIn
    userId = row[0]
    businessId = row[1]

    if userId in userAttributesRddDict:
        userReviewCount = userAttributesRddDict[userId][0]
        userAvgStars = userAttributesRddDict[userId][1]

    if businessId in businessAttrDict:
        busReviewCount = businessAttrDict[businessId][1]
        busStars = businessAttrDict[businessId][0]

    if businessId in checkAttributesDict:
        checkIn = checkAttributesDict[businessId]

    #return (usersDictionary[userId], businessDictionary[businessId], userReviewCount, userAvgStars, busReviewCount, busStars, checkIn)
    return (userReviewCount, userAvgStars, busReviewCount, busStars, checkIn)

def to_labeled_point(sc, features, labels, categorical=False):
    """Convert numpy arrays of features and labels into
    a LabeledPoint RDD for MLlib and ML integration.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :param categorical: boolean, whether labels are already one-hot encoded or not
    :return: LabeledPoint RDD with features and labels
    """
    labeled_points = []
    for x, y in zip(features, labels):
        if categorical:
            lp = LabeledPoint(y, x)
        else:
            lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return sc.parallelize(labeled_points)


trainInput = trainDataRdd.map(lambda x: featureCombine(x)).collect()
trainRating = trainData.map(lambda x: float(x[2])).collect()
testInput = testEachDataRdd.map(lambda x: featureCombine(x)).collect()
testRating = testData.map(lambda x: float(x[2])).collect()
testMapUserBusiness = testData.map(lambda x : ((usersDictionary[x[0]],businessDictionary[x[1]]),(x[0],x[1]))).collectAsMap()

trainInputArray = np.array(trainInput)
trainInputLabel = np.array(trainRating)
testInputArray = np.array(testInput)
testInputLabel = np.array(testRating)

#Convert to labeled points
labeledTrainRdd = to_labeled_point(sc,trainInputArray,trainInputLabel)
(trainingData, testData1) = labeledTrainRdd.randomSplit([0.7, 0.3])
# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
model = RandomForest.trainClassifier(trainingData, numClasses=6, categoricalFeaturesInfo={},
                                     numTrees=500, featureSubsetStrategy="sqrt",
                                     impurity='gini', maxDepth=5, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData1.map(lambda x: x.features))
labelsAndPredictions = testData1.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(testData1.count())
testRmse = labelsAndPredictions.map(lambda x:(x[1]-x[0])*(x[1]-x[0])).mean()
print("RMSE :"+str(math.sqrt(testRmse)))
#print('Test Error = ' + str(testErr))
#print('Learned classification forest model:')
#print(model.toDebugString())

print("Trying on validation set ")
#labeled points
labeledTestRdd = to_labeled_point(sc,testInputArray,testInputLabel)

validPred = model.predict(labeledTestRdd.map(lambda x:x.features))
labelsAndPredictionsValid = labeledTestRdd.map(lambda lp: lp.label).zip(validPred)
validRmse = labelsAndPredictionsValid.map(lambda x:(x[1]-x[0])*(x[1]-x[0])).mean()
print("RMSE :"+str(math.sqrt(validRmse)))

print(time.time() - start)

ratesAndPredsVal = labelsAndPredictionsValid.collect()
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
for val in ratesAndPredsVal:
    if abs(val[0] - val[1]) <= 1 and  abs(val[0] - val[1]) > 0:
        count1 = count1 + 1
    elif abs(val[0] - val[1]) <= 2 and abs(val[0] - val[1]) >1:
        count2 = count2 + 1
    elif abs(val[0] - val[1]) <= 3 and abs(val[0] - val[1]) > 2:
        count3 = count3 + 1
    elif abs(val[0] - val[1]) <= 4 and abs(val[0] - val[1]) > 3:
        count4 = count4 + 1
    elif abs(val[0] - val[1]) <= 5 and abs(val[0] - val[1]) > 4:
        count5 = count5 + 1


print("1 diff: "+str(count1))
print("2 diff: "+str(count2))
print("3 diff: "+str(count3))
print("4 diff: "+str(count4))
print("5 diff: "+str(count5))
