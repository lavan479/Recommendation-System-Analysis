from pyspark import SparkConf, SparkContext
import time
import json
import sys
import math
import itertools
import numpy as np
import xgboost as xgb
sc = SparkContext('local[*]', 'HybridCF')
sc.setLogLevel("ERROR")

########################################################################################
#MODEL BASED SYSTEM

start = time.time()
mlModelDict = {}
itemCFModelDict = {}

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
userAttributesRddDict = userRdd.map(lambda x:(x['user_id'],(x['review_count'],x['average_stars'],x['useful']
                                                            ,x['funny'],x['cool'],x['fans'],x['compliment_profile']))).collectAsMap()
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
    global userReviewCount, userAvgStars, busReviewCount, busStars, checkIn
    userful = 0
    funny = 0
    cool = 0
    compliments = 0
    fans = 0
    if userId in userAttributesRddDict:
        userReviewCount = userAttributesRddDict[userId][0]
        userAvgStars = userAttributesRddDict[userId][1]
        userful = userAttributesRddDict[userId][2]
        funny = userAttributesRddDict[userId][3]
        cool = userAttributesRddDict[userId][4]
        compliments = userAttributesRddDict[userId][5]
        fans = userAttributesRddDict[userId][6]

    if businessId in businessAttrDict:
        busReviewCount = businessAttrDict[businessId][1]
        busStars = businessAttrDict[businessId][0]

    if businessId in checkAttributesDict:
        checkIn = checkAttributesDict[businessId]

    return (usersDictionary[userId], businessDictionary[businessId],
            userReviewCount, userAvgStars, busReviewCount, busStars,
            checkIn, userful, funny, cool, compliments, fans)

###############################################################################################################################
trainInput = trainDataRdd.map(lambda x: featureCombine(x)).collect()
trainRating = trainData.map(lambda x: float(x[2])).collect()
testInput = testEachDataRdd.map(lambda x: featureCombine(x)).collect()
testRating = testData.map(lambda x: float(x[2])).collect()
testMapUserBusiness = testData.map(lambda x : ((usersDictionary[x[0]],businessDictionary[x[1]]),(x[0],x[1]))).collectAsMap()
###################################################################################
model = xgb.XGBRegressor(n_estimators=1000)
inputArray = np.array(trainInput)
inputRating = np.array(trainRating)
testInputArray = np.array(testInput)
testInputRatingArray = np.array(testRating)

###################################################################################

model.fit(inputArray[:,2:],inputRating)
testPred = model.predict(testInputArray[:,2:])

for i in range(len(testInputArray)):
    val = (testInputArray[i][0],testInputArray[i][1])
    mlModelDict[testMapUserBusiness[val]] = testPred[i]

###############################################################################################
pearsonCoefficientMatrix = {}
usersDictionary = {}
glo_list_renew =[[9, 101, 193], [1, 901, 24593], [3, 23, 769],[33, 803, 49157],
              [13, 552, 98317], [17, 37, 3079], [19, 63, 97], [27, 67, 6151],
              [29, 29, 12289], [11, 91, 1543],[31, 79, 53], [37, 119, 389],
            [39, 552, 98317], [59, 119, 389], [41, 37, 3079],[47, 67, 6151],
            [51, 29, 12289], [53, 79, 53], [57, 803, 49157],  [43, 63, 97] ]

numOfHash = 20
numOfRows = 2

def convertToIndex(row):
    listOfVals = []
    for val in row:
        listOfVals.append(usersDictionary[val])
    return listOfVals

def getSignatureUsingMinHash(row):
    global m
    minSignature = [min(((ax[0] * x + ax[1] + 7) % ax[2]) % m for x in row) for ax in glo_list_renew]
    return minSignature

def calculateJaccardSimilarity(vec1, vec2):
    set1 = set(vec1)
    set2 = set(vec2)
    return float(len(set1 & set2) / len(set1 | set2))

def generateSmallBands(signatureData):
    index = 0
    bandSize = int(len(signatureData[1]) / numOfRows)
    business_id = signatureData[0]
    signaturesList = signatureData[1]
    bandList = []
    for band in range(bandSize):
        rowList = []
        for row in range(numOfRows):
            rowList.append(signaturesList[index])
            index = index+1
        bandList.append(((band, tuple(rowList)), [business_id]))
        rowList.clear()

    return bandList

###############################################################################################

def getNormaliizedRating(x):
    business = x[0]
    ratings = x[1]
    ratingList = []
    for rate in ratings:
        ratingList.append(float(rate))
    avgRating = sum(ratingList)
    avgRating = avgRating/len(ratingList)
    return business, (ratingList, avgRating)

def getTupleInUserBusIdRatingAvg(listOfVals):
    rows = []
    for val in range(0, len(listOfVals[1][0])):
        rows.append(((listOfVals[0][1][val], listOfVals[0][0]), (listOfVals[1][0][val], listOfVals[1][1])))
    return rows

def pearsonSimilarity(businessPair, businessToUserRatingAvg,userBusToRatingAvg):

    business1 = businessPair[0]
    business2 = businessPair[1]
    if business1 in businessToUserRatingAvg and business2 in businessToUserRatingAvg:
        usersForBusiness1 = businessToUserRatingAvg[business1][0]
        usersForBusiness2 = businessToUserRatingAvg[business2][0]
        #Using co-rated
        coRatedUsers = set(usersForBusiness1) & set(usersForBusiness2)
        if len(coRatedUsers) == 0:
            return 0
        magnitudeForVector1 = 0
        magnitudeForVector2 = 0
        dotProductOfUsers = 0

        for user in coRatedUsers:
               rate1 = float(userBusToRatingAvg[(user,business1)][0]) - float(userBusToRatingAvg[(user,business1)][1])
               rate2 = float(userBusToRatingAvg[(user, business2)][0]) - float(userBusToRatingAvg[(user, business2)][1])
               magnitudeForVector1 = magnitudeForVector1 + rate1*rate1
               magnitudeForVector2 = magnitudeForVector2 + rate2*rate2
               dotProductOfUsers = dotProductOfUsers + rate1*rate2

        global pearsonCoefficientMatrix
        if magnitudeForVector1*magnitudeForVector2 == 0:
            similarity = 0
            pearsonCoefficientMatrix[businessPair] = similarity
            return similarity
        similarity = float(dotProductOfUsers/math.sqrt(magnitudeForVector1*magnitudeForVector2))
        pearsonCoefficientMatrix[businessPair] = similarity
    return pearsonCoefficientMatrix[businessPair]

def pearsonPrediction(userId, userBusToRatingAvg, topCosineSimilarities):

    predictedVal = 0
    denominatorAbsSum = 0
    for eachBusinessVal in topCosineSimilarities:
        val = eachBusinessVal[1]
        denominatorAbsSum = denominatorAbsSum + abs(val)
        predictedVal = predictedVal + float(userBusToRatingAvg[(userId,eachBusinessVal[0])][0])*val

    if float(denominatorAbsSum) == float(0):
        return 0
    predict = float(predictedVal/denominatorAbsSum)
    if predict > 5:
        predict = 5
    elif predict < 0:
        predict = 0

    return predict
# Need to handle cold start problem
def findActualSimilarBusinesses(businessSimilarityList):
    N = 200
    sortedSimilarityList = sorted(businessSimilarityList,key=(lambda x:-x[1]))
    if N >= len(sortedSimilarityList):
        return sortedSimilarityList
    newList = sortedSimilarityList[:N]
    return newList

def getItemRatings(userBusinessPair, usersToBusiness, businessToUserRatingAvg, userBusToRatingAvg,jaccardSimilarity):

    userId = userBusinessPair[0]
    businessId = userBusinessPair[1]
    ratedBusinesses = usersToBusiness[userId]
    pearsonSimilarityList = []
    similarity = 0
    if userId not in usersToBusiness and businessId not in businessToUserRatingAvg:
        predictedRating = 3.0
        return ((userBusinessPair), predictedRating)

    elif userId not in usersToBusiness:
        predictedRating = float(businessToUserRatingAvg[businessId][2])
        return ((userBusinessPair), predictedRating)

    elif businessId not in businessToUserRatingAvg:
        averageUserRating = 0
        for businessRatedByUser in ratedBusinesses:
            averageUserRating = averageUserRating + userBusToRatingAvg[(userId, businessRatedByUser)][0]
        averageUserRating = float(averageUserRating / len(ratedBusinesses))
        predictedRating = averageUserRating
        return ((userBusinessPair), predictedRating)

    elif (userBusinessPair) in userBusToRatingAvg:

        return ((userBusinessPair), userBusToRatingAvg[(userBusinessPair)][0])

    for business in ratedBusinesses:
        if business != businessId:
          if (businessId, business) not in jaccardSimilarity.keys() and (business, businessId) not in jaccardSimilarity.keys():
                continue
          elif (businessId,business) in pearsonCoefficientMatrix:
                similarity = pearsonCoefficientMatrix[(businessId, business)]
          elif (business,businessId) in pearsonCoefficientMatrix:
                similarity = pearsonCoefficientMatrix[(business, businessId)]
          elif business in businessToUserRatingAvg and businessId in businessToUserRatingAvg:
                similarity = pearsonSimilarity((businessId,business),businessToUserRatingAvg,userBusToRatingAvg)
        if similarity > 0:
           pearsonSimilarityList.append((business,similarity))

    sortedList = findActualSimilarBusinesses(pearsonSimilarityList)
    if len(sortedList) == 0:
        averageUserRating = 0
        for businessRatedByUser in ratedBusinesses:
            averageUserRating = averageUserRating + userBusToRatingAvg[(userId, businessRatedByUser)][0]
        averageUserRating = float(averageUserRating / len(ratedBusinesses))
        predictedRating = averageUserRating
        return ((userBusinessPair), predictedRating)

    predictedRating = pearsonPrediction(userId,userBusToRatingAvg,sortedList)
    return ((userBusinessPair), predictedRating)

def writeToFile(outputRdd):
    toWriteFile = open(outputFile, 'w')
    toWriteFile.write("user_id, business_id, prediction")
    for i in outputRdd.collect():
        toWriteFile.write("\n")
        toWriteFile.write(i[0][0] + "," + i[0][1] + "," + str(i[1]))
    toWriteFile.close()

########################### MAIN STARTS ##########################################
textRdd = sc.textFile(inputTrainFile)

trainData = textRdd.map(lambda row: row.split(',')).filter(lambda x: x[0] != "user_id").persist()
completeRdd = trainData.map(lambda vals: (vals[1], vals[0]))
userIds = completeRdd.map(lambda x:x[1]).distinct().collect()
userRows = len(userIds)

for u in range(0, userRows):
    usersDictionary[userIds[u]] = u
#Creating Characterstric matrix
charactersticMatrix = completeRdd.groupByKey().mapValues(convertToIndex)
charactersticDict = charactersticMatrix.collectAsMap()
m = userRows
#Creating signature Matrix
signatureMatrix = charactersticMatrix.mapValues(getSignatureUsingMinHash)
signatureDict = signatureMatrix.collectAsMap()

#Applying LSH
smallSignatureBands = signatureMatrix.flatMap(lambda x: generateSmallBands(x))
candidates = smallSignatureBands.reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) > 1)
candidatePairs = candidates.flatMap(lambda x: list(itertools.combinations(x[1],2))).distinct()

allJaccardSimilarity = candidatePairs.map( lambda x: (tuple(sorted([x[0],x[1]])),calculateJaccardSimilarity(charactersticDict[x[0]], charactersticDict[x[1]])))
jaccardSimilarity = allJaccardSimilarity.filter(lambda x: (x[1] >= 0.5)).distinct().collectAsMap()

testData = sc.textFile(inputTestFile)
testData = testData.map(lambda row: row.split(',')).filter(lambda x: x[0] != "user_id").persist()
testEachDataRdd = testData.map(lambda x: (x[0], x[1]))

#meanOfRatings = trainData.map(lambda x:( float(x[2]))).mean()
usersToBusiness = trainData.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a,b:a+b).collectAsMap()  #To find all businesses for given user
businessToUsers = trainData.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda a,b:a+b)   #b,{users}
businessRatings = trainData.map(lambda x: (x[1], [x[2]])).reduceByKey(lambda a,b:a+b)
normalizedRating = businessRatings.map(lambda x: getNormaliizedRating(x))  #b{rate}

businessUsersRatingAvg = businessToUsers.join(normalizedRating).map(lambda x: ((x[0], x[1][0]), (x[1][1][0], x[1][1][1])))
userBusToRatingAvg = businessUsersRatingAvg.flatMap(lambda x: getTupleInUserBusIdRatingAvg(x)).collectAsMap() #u[i],bid, rating, avg
businessToUserRatingAvg = businessUsersRatingAvg.map(lambda x: (x[0][0], (x[0][1], x[1][0], x[1][1]))).collectAsMap()

predictedRatings = testEachDataRdd.map(lambda x: getItemRatings(x, usersToBusiness, businessToUserRatingAvg, userBusToRatingAvg, jaccardSimilarity))
itemCFModelDict = predictedRatings.collectAsMap()
######################################################################

def findHybridRating(userBusinessPair):
    userId = userBusinessPair[0]
    businessId = userBusinessPair[1]
    ratedBusinesses = usersToBusiness[userId]
    pearsonSimilarityList = []
    #Assigning weights
    if userId not in usersToBusiness and businessId not in businessToUserRatingAvg:
        predictedRating = itemCFModelDict[userBusinessPair] * 0 + mlModelDict[userBusinessPair] * 1.0
        return ((userBusinessPair), predictedRating)

    elif userId not in usersToBusiness:
        predictedRating = itemCFModelDict[userBusinessPair] * 0.2 + mlModelDict[userBusinessPair] * 0.8
        return ((userBusinessPair), predictedRating)

    elif businessId not in businessToUserRatingAvg:
        predictedRating = itemCFModelDict[userBusinessPair] * 0.2 + mlModelDict[userBusinessPair] * 0.8
        return ((userBusinessPair), predictedRating)

    elif (userBusinessPair) in userBusToRatingAvg:
        return ((userBusinessPair),itemCFModelDict[userBusinessPair])
    similarity = 0
    for business in ratedBusinesses:
        if business != businessId:
            if (businessId, business) not in jaccardSimilarity.keys() and (
            business, businessId) not in jaccardSimilarity.keys():
                continue
            elif (businessId, business) in pearsonCoefficientMatrix:
                similarity = pearsonCoefficientMatrix[(businessId, business)]
            elif (business, businessId) in pearsonCoefficientMatrix:
                similarity = pearsonCoefficientMatrix[(business, businessId)]
        if similarity > 0:
            pearsonSimilarityList.append((business, similarity))
    sortedList = findActualSimilarBusinesses(pearsonSimilarityList)
    if len(sortedList) == 0:
        predictedRating = itemCFModelDict[userBusinessPair]*0.2 + mlModelDict[userBusinessPair]*0.8
        return ((userBusinessPair), predictedRating)

    if len(sortedList) > 5 and userAttributesRddDict[userBusinessPair[0]][0] < 10:
        predictedRating = itemCFModelDict[userBusinessPair] * 0.7 + mlModelDict[userBusinessPair] * 0.3

    elif len(sortedList) > 5 and userAttributesRddDict[userBusinessPair[0]][0] > 10 and userAttributesRddDict[userBusinessPair[0]][0] < 100:
        predictedRating = itemCFModelDict[userBusinessPair] * 0.5 + mlModelDict[userBusinessPair] * 0.5
    else:
        predictedRating = itemCFModelDict[userBusinessPair] * 0.3 + mlModelDict[userBusinessPair] * 0.7

    return (userBusinessPair), predictedRating


finalPredictionValues = testEachDataRdd.map(lambda x: (findHybridRating(x)))
writeToFile(finalPredictionValues)
ratesAndPreds = testData.map(lambda x: ((x[0], x[1]), float(x[2]))).join(finalPredictionValues)
MSE = ratesAndPreds.map(lambda r: ((r[1][0] - r[1][1])*(r[1][0] - r[1][1]))).mean()
print("Root Mean Squared Error = " + str(math.sqrt(MSE)))
print("Hybrid")
print("Duration: "+str(time.time() - start))
