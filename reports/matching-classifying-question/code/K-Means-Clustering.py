from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.feature import Word2Vec
from numpy import array
from math import sqrt
import csv
import re

conf = SparkConf().setAppName("Question Matching")
sc = SparkContext(conf=conf)

#Create an SQL context for the CSV
sqlContext = SQLContext(sc)
df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").option("delimiter", '\t').load("hdfs:///shared/stackOverflow/data_formatted.csv")

data = df.rdd

TAKEN = data.count()

#Clean the stackoverflow data using regex
cleanData = data.map(lambda x: re.sub(r"(&lt;)|(p&gt;)|(&#xA;)|(/)|(&gt;)|(&quot;)|(&amp;)|(nbsp;)|(lt;)"," ", x[6])).filter(lambda x: len(x)>10).repartition(2000)

dtNow = cleanData.map(lambda x: x.split())

#Create the word2vec model
word2vec = Word2Vec().setVectorSize(3).setSeed(42).setNumPartitions(200)
model = word2vec.fit(dtNow)

#Get all the vectors into a dictionary
vectors_ = model.getVectors() 
vectors = {k: DenseVector([x for x in vectors_.get(k)])
    for k in vectors_.keys()}

'''
Turn the sentences into vectors by adding each word specific vector.

sentence : str
	the sentence to be turned into a vector
'''
def sentence2vec(sentence):
	sentenceVector = [0]*3
	for word in sentence:
		if word in vectors:
			for num in range(len(sentenceVector)):
				try:
					sentenceVector[num] += tuple(vectors[word])[num]
				except:
					sentenceVector[num] += 0
	return sentenceVector


'''
Find the squared magnitude of a given vector

vector : List [int, int, int]
	vector to find the squared distance to
'''
def euclidDistance(vector):
    result = 0
    for num in vector:
        result += num**2
    return result

'
Find the cosine similarity between a given vector and the vector [1,1,1]

vector: List [int,int,int]
	vector to find the cosine similarity to
	
return
	Both the magnitude of the vector and the cosine similarity
'''
def cosineSimilarity(vector):
    result = 0
    for num in vector:
        result += num
        magnitude = sqrt(euclidDistance(vector))
        answer = 0
        if magnitude != 0:
            answer = result/(magnitude*1.73)
    return euclidDistance(vector), answer


parsed = dtNow.map(sentence2vec).map(cosineSimilarity)

#Cluster the vectors with a certain K value
clusters = KMeans.train(parsed, TAKEN//2, maxIterations=10, runs=1, initializationMode="random")

def error(point):
	center = clusters.centers[clusters.predict(point)]
	return sqrt(sum([x**2 for x in (point - center)]))


WSSSE = par.map(error).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

#WSSSE - K number
#2748952517804.801 - 15
#845733672280.8569 - 200
#84410322136.5304 - 5000
#24293625773.70967 - 25,000


               
               

