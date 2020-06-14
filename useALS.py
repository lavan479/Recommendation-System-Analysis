from pyspark import SparkConf, SparkContext
import time
import json
import sys
import math
import numpy as np
import xgboost as xgb
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sklearn as sk
import sys
sc = SparkContext('local[*]', 'ModelCF')
sc.setLogLevel("ERROR")

start = time.time()

inputTrainFile = sys.argv[1]
inputUserFile = sys.argv[2]
inputTipFile = sys.argv[3]
inputCheckInFile = sys.argv[4]
inputBusinessFile = sys.argv[5]
inputTestFile = sys.argv[6]
outputFile =sys.argv[7]
########################################################################################
#Input file RDD

trainData = sc.textFile(inputTrainFile)
trainData = trainData.map(lambda row: row.split(',')).filter(lambda x: x[0] != "user_id").persist()
trainDataRdd = trainData.map(lambda x:(x[0],x[1],x[2]))
testData = sc.textFile(inputTestFile)
testData = testData.map(lambda row: row.split(',')).filter(lambda x: x[0] != "user_id").persist()
testEachDataRdd = testData.map(lambda x: (x[0], x[1]))

#########################################################################################
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

def writeToFile(outputRddMap, testRating, combineTestFeature):
    toWriteFile = open(outputFile, 'w')
    toWriteFile.write("user_id, business_id, prediction")
    for i in range(len(combineTestFeature)):
        toWriteFile.write("\n")
        val1 = outputRddMap[(combineTestFeature[i][0],combineTestFeature[i][1])][0]
        val2 = outputRddMap[(combineTestFeature[i][0],combineTestFeature[i][1])][1]
        toWriteFile.write(val1 + "," + val2 + "," + str(testRating[i]))
    toWriteFile.close()
###############################################################################################################################

trainInput = trainDataRdd.map(lambda x: (usersDictionary[x[0]],businessDictionary[x[1]],float(x[2])))
trainRating = trainData.map(lambda x: float(x[2])).collect()

testInput = testEachDataRdd.map(lambda x: (usersDictionary[x[0]],businessDictionary[x[1]]))
testInputWithRating = testData.map(lambda x: ((usersDictionary[x[0]],businessDictionary[x[1]]),float(x[2])))

###################################################################################

def cutPred(x):

    if x[2] > 5:
        rate = 5
    elif x[2] < 1:
        rate = 1
    else:
        rate = x[2]
    return ((x[0], x[1]), rate)


rank = 3
numIterations = 20
model = ALS.train(trainInput, rank, numIterations,0.23)
preds = model.predictAll(testInput).map(cutPred)

ratesAndPreds = testInputWithRating.join(preds)
MSE = ratesAndPreds.map(lambda r: ((r[1][0] - r[1][1])*(r[1][0] - r[1][1]))).mean()
print("Root Mean Squared Error = " + str(math.sqrt(MSE)))
print("Duration :"+str(time.time()-start))

ratesAndPredsVal = ratesAndPreds.collect()
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
for val in ratesAndPredsVal:
    if abs(val[1][0] - val[1][1]) <= 1 and  abs(val[1][0] - val[1][1]) > 0:
        count1 = count1 + 1
    elif abs(val[1][0] - val[1][1]) <= 2 and abs(val[1][0] - val[1][1]) >1:
        count2 = count2 + 1
    elif abs(val[1][0] - val[1][1]) <= 3 and abs(val[1][0] - val[1][1]) > 2:
        count3 = count3 + 1
    elif abs(val[1][0] - val[1][1]) <= 4 and abs(val[1][0] - val[1][1]) > 3:
        count4 = count4 + 1
    elif abs(val[1][0] - val[1][1]) <= 5 and abs(val[1][0] - val[1][1]) > 4:
        count5 = count5 + 1

print("1 diff: "+str(count1))
print("2 diff: "+str(count2))
print("3 diff: "+str(count3))
print("4 diff: "+str(count4))
print("5 diff: "+str(count5))
