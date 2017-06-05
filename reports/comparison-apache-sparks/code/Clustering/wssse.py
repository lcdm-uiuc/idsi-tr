from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
import json
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("KMeans WSSSE")
sc = SparkContext(conf=conf)


coordinates = sc.textFile("hdfs:///user/emilojkovic/data/az_businesses_kmeans/part-00000")

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

errors = []
# Build the model (cluster the data)
for i in range(1, 15):
    clusters = KMeans.train(data, i, maxIterations=300, runs=10, initializationMode="k-means")
    WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    errors.append((i, str(WSSSE)))

sc.parallelize(errors).coalesce(1).saveAsTextFile('hdfs:///user/emilojkovic/kmeans_wssse')
