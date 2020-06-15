from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
import time
import json
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers
sc = SparkContext('local[*]')
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

####################################################################################################

trainInput = trainDataRdd.map(lambda x: featureCombine(x)).collect()
trainRating = trainData.map(lambda x: float(x[2])).collect()
testInput = testEachDataRdd.map(lambda x: featureCombine(x)).collect()
testRating = testData.map(lambda x: float(x[2])).collect()
testMapUserBusiness = testData.map(lambda x : ((usersDictionary[x[0]],businessDictionary[x[1]]),(x[0],x[1]))).collectAsMap()

trainInputArray = np.array(trainInput)
trainInputLabel = np.array(trainRating)
testInputArray = np.array(testInput)
testInputRatingArray = np.array(testRating)

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

#Convert to labeled points
labeledTrainRdd = to_labeled_point(sc,trainInputArray,trainInputLabel)
(trainingDataAfterSplit, validDataAfterSplit) = labeledTrainRdd.randomSplit([0.7, 0.3])

trainInputArrayAfterSplit = np.array(trainingDataAfterSplit.map(lambda x: x.features))
trainInputLabelAfterSplit = np.array(trainingDataAfterSplit.map(lambda x: x.label))
validInputDataAfterSplit = np.array(validDataAfterSplit.map(lambda x: x.features))
validInputLabelAfterSplit = np.array(validDataAfterSplit.map(lambda x: x.label))

####################################################################################################
#MODEL

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))#kernel_initializer=tf.keras.regularizers.l2(0.1)))
# Add another:
model.add(layers.Dense(64, activation='relu'))#, kernel_initializer=tf.keras.regularizers.l2(0.1)))
#Add another:
model.add(layers.Dense(64, activation='relu'))#,kernel_initializer=tf.keras.regularizers.l2(0.1)))
#Add another:
#model.add(layers.Dense(64, activation='relu',kernel_initializer='ones'))
# Add an output layer with 10 output units:
model.add(layers.Dense(6))

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.00001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.fit(trainInputArrayAfterSplit, trainInputLabelAfterSplit, epochs=10, batch_size=32)
model.fit(trainInputArray, trainInputLabel, epochs=10, batch_size=32)

result = model.predict(testInputArray, batch_size=32)
print(result.shape)
rmse = 0
for i in range(len(result)):
    label = np.argmax(result[i],axis=0)
    val = abs(testInputRatingArray[i] - label)
    rmse = rmse + val*val

finalRmse = math.sqrt(float(rmse/len(result)))

print("RMSE :"+str(finalRmse))
print(str(time.time() - start))

