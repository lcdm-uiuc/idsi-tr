import sys
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
import pandas as pd
import csv
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

# Remove the first line from the csv file
def clean(x):
    if (x[29] != "Amount"):
        return x

#Turn the data into a labeled point using 30 dimensions
def normalize(x):
    return LabeledPoint(float(x[30]), [float(x[0]), float(x[29])/ 25691.16])

sameModel = DecisionTreeModel.load(sc, "./decisiontreefraud")

#make a spark conference
conf = (SparkConf()
     .setMaster("local")
     .setAppName("My app")
     .set("spark.executor.memory", "4g"))

#files have to be added while running to see the data in the stream
ssc = StreamingContext(sc, 1)
lines1 = ssc.textFileStream("file:///mnt/vdatanodea/datasets/creditcards/credit/b")
trainingData = lines1.map(lambda line: LabeledPoint( float(line.split(" ")[1]), [ (line.split(" ") [0]) ,  (line.split(" ") [2]) ])).cache()
trainingData.pprint()

lines2 = ssc.textFileStream("file:///mnt/vdatanodea/datasets/creditcards/credit/c")
testData = lines2.map(lambda line: LabeledPoint( float(line.split(" ")[1]), [ (line.split(" ") [0]) ,  (line.split(" ") [2]) ])).cache()
testData.pprint()

#print out the predicted values
def handle_rdd(rdd):
    for r in rdd.collect():
        print( r.map(lambda p: (p.label, p.features, sameModel.predict(p.features))) )

labelsAndPreds = testData.transform(lambda rdd: handle_rdd)
labelsAndPreds.pprint()
ssc.start()
