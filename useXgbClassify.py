from pyspark import SparkConf, SparkContext
import time
import json
import sys
import math
import numpy as np
import xgboost as xgb
import sklearn as sk
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
    userId = row[0]
    businessId = row[1]
    checkIn = 0
    userReviewCount = 0
    userAvgStars = 0
    busReviewCount = 0
    busStars = 0
    if userId in userAttributesRddDict:
        userReviewCount = userAttributesRddDict[userId][0]
        userAvgStars = userAttributesRddDict[userId][1]

    if businessId in businessAttrDict:
        busReviewCount = businessAttrDict[businessId][1]
        busStars = businessAttrDict[businessId][0]

    if businessId in checkAttributesDict:
        checkIn = checkAttributesDict[businessId]
    return ( userReviewCount, userAvgStars, busReviewCount, busStars,checkIn)

#########################################################################################
def writeToFile(outputRddMap, testRating, combineTestFeature):
    toWriteFile = open(outputFile, 'w')
    toWriteFile.write("user_id, business_id, prediction")
    for i in range(len(combineTestFeature)):
        toWriteFile.write("\n")
        val1 = outputRddMap[(combineTestFeature[i][0],combineTestFeature[i][1])][0]
        val2 = outputRddMap[(combineTestFeature[i][0],combineTestFeature[i][1])][1]
        toWriteFile.write(val1 + "," + val2 + "," + str(testRating[i]))
    toWriteFile.close()
#########################################################################################

trainInput = trainDataRdd.map(lambda x: featureCombine(x)).collect()
trainRating = trainData.map(lambda x: float(x[2])).collect()
testInput = testEachDataRdd.map(lambda x: featureCombine(x)).collect()
testRating = testData.map(lambda x: float(x[2])).collect()
testMapUserBusiness = testData.map(lambda x : ((usersDictionary[x[0]],businessDictionary[x[1]]),(x[0],x[1]))).collectAsMap()

#########################################################################################
model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', num_class=6,
       max_depth=5, n_estimators=600, learning_rate=0.01,
       n_jobs=-1, objective='multi:softmax')

inputArray = np.array(trainInput)
inputRating = np.array(trainRating)
testInputArray = np.array(testInput)
testInputRatingArray = np.array(testRating)

model.fit(inputArray,inputRating)
testPred = model.predict(testInputArray)
mse = sk.metrics.mean_squared_error(testRating, testPred)

print("RMSE :"+str(math.sqrt(mse)))
print("Duration: "+str(time.time() - start))

count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
for i in range(len(testPred)):
    val0 = testPred[i]
    val1 = testRating[i]
    if abs(val0 - val1) <= 1 and  abs(val0 - val1) > 0:
        count1 = count1 + 1
    elif abs(val0 - val1) <= 2 and abs(val0 - val1) >1:
        count2 = count2 + 1
    elif abs(val0 - val1) <= 3 and abs(val0 - val1) > 2:
        count3 = count3 + 1
    elif abs(val0 - val1) <= 4 and abs(val0 - val1) > 3:
        count4 = count4 + 1
    elif abs(val0 - val1) <= 5 and abs(val0 - val1) > 4:
        count5 = count5 + 1

print("1 diff: "+str(count1))
print("2 diff: "+str(count2))
print("3 diff: "+str(count3))
print("4 diff: "+str(count4))
print("5 diff: "+str(count5))
