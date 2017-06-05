from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
import json
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("KMeans Artificial Clustering")
sc = SparkContext(conf=conf)

def map_to_lat_lng(string):
    string = string.replace("(", "").replace(")", "").replace(",", "").split()
    if len(string) == 2:
       a = float(string[0])
       b = float(string[1])
       return (a, b)
    return None

data = sc.textFile("hdfs:///user/emilojkovic/clusters.txt")
data = data.map(map_to_lat_lng)

clusters = KMeans.train(data, 7, maxIterations=300, runs=10, initializationMode="k-means")
point = data.map(lambda x: (x, clusters.predict(x)))

point.coalesce(1).saveAsTextFile('hdfs:///user/emilojkovic/mllib_kmeans_artificial')
#sc.parallelize(point).coalesce(1).saveAsTextFile('hdfs:///user/emilojkovic/mllib_kmeans')
