from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.classification import NaiveBayes, LogisticRegressionWithSGD
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
import csv
import string
import nltk
from gensim.models import word2vec
from numpy import add
import re
import numpy as np
from collections import Counter
from scipy.spatial.distance import *
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

conf = SparkConf().setAppName("project")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

def clean_sentence(line):
    cleanedString = ""
    line = re.sub(r'[^\w\s]','',line)
    for letter in line:
        if ord(letter) > 127:
            cleanedString += str(ord(letter))
        else:
            cleanedString += letter.lower()
    return cleanedString


def weight_vector_tfidf(weights, sentence):
    vector = []
    for word in sentence:
        if word is not '':
            tfidfScore = weights.get(word)
            vector.append(tfidfScore)
    return vector


def weight_vector_w2v(w2v, sentence):
    vector = []
    for word in sentence:
        if word is not '':
            w2vVector = w2v.wv[word]
            vector.append(w2vVector)
    return vector


def weight_vector_both(weights, w2v, sentence):
    vector = []
    for word in sentence:
        if word is not '':
            w2vVector = w2v.wv[word]
            tfidfScore = weights.get(word)
            vector.append(w2vVector*tfidfScore)
    return vector


def get_weight(count, eps=10000, min_count=1):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def tfidf(corpus):
    eps = 5000
    words = [x for sublist in corpus for x in sublist]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    return weights


def jaccard_index(sentenceA, sentenceB):
    sentenceA = set(sentenceA)
    sentenceB = set(sentenceB)
    setIntersection = sentenceA.intersection(sentenceB)
    return float(len(setIntersection)) / (len(sentenceA) + len(sentenceB) - len(setIntersection))


def get_sum(array):
    total = 0
    for vector in array:
        total += vector
    return total


def get_mean(array):
    total = 0
    for vector in array:
        total += vector
    if len(array) != 0:
        return total / len(array)
    return total


def sum_w2v(sentence):
    newVector = [0]*100
    for vector in sentence:
        for i in range(len(vector)):
            newVector[i] += vector[i]
    newVector = np.array(newVector)
    return newVector


def mean_w2v(sentence):
    newVector = [0]*100
    for vector in sentence:
        for i in range(len(vector)):
            newVector[i] += vector[i]
    newVector = np.array(newVector)
    return newVector / len(sentence)

def get_cosine(pair_of_vectors):
    try:
        cosineScore = cosine(pair_of_vectors[0],pair_of_vectors[1])
    except Exception as e:
        cosineScore = 1.0
    return cosineScore


def get_results(data):
    results = []
    for i in data:
        splits = i.randomSplit([0.8, 0.2], 1234)
        training = splits[0]
        testing = splits[1]
        training = training.toDF(["label", "features"])
        testing = testing.toDF(["label", "features"])
        numFeatures = training.take(1)[0].features.size

        #First layers has to be the number of the features of the data

        layers = [numFeatures, 4, 5, 2]

        trainer = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=128, seed=1234)
        model = trainer.fit(training)
        result = model.transform(testing)
        predictionAndLabels = result.select("prediction", "label")
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        answer = "Test set precision = " + str(evaluator.evaluate(predictionAndLabels)) + '\n'
        results.append(answer)
    return sc.parallelize(results)


def get_RDDs(data, corpus, weights, questions, labels):
    w2v = word2vec.Word2Vec(corpus, size=100, window=20, min_count=1, workers=40)
    one = questions.map(lambda x : (weight_vector_tfidf(weights, x[0]), weight_vector_tfidf(weights, x[1])))
    one_sum = one.map(lambda x : (get_sum(x[0]), get_sum(x[1]) ))
    one_mean = one.map(lambda x : (get_mean(x[0]), get_mean(x[1])))
    two = questions.map(lambda x : (weight_vector_w2v(w2v, x[0]), weight_vector_w2v(w2v, x[1])))
    two_sum = two.map(lambda x : (sum_w2v(x[0]), sum_w2v(x[1])))
    two_mean = two.map(lambda x : (mean_w2v(x[0]), mean_w2v(x[1])))
    three = questions.map(lambda x : jaccard_index(x[0], x[1]))
    four = questions.map(lambda x : (weight_vector_both(weights, w2v, x[0]), weight_vector_both(weights, w2v, x[1])))
    four_sum = four.map(lambda x : (sum_w2v(x[0]), sum_w2v(x[1])))
    four_mean = four.map(lambda x : (mean_w2v(x[0]), mean_w2v(x[1])))
    five_sum = four_sum.zip(three)
    five_mean = four_mean.zip(three)
    labels = labels.coalesce(1)
    one_sum_difference = labels.zip(one_sum.map(lambda x : abs(x[0]-x[1])).coalesce(1)).repartition(100).map(lambda x :  (x[0], Vectors.dense(x[1])))
    one_mean_difference  = labels.zip(one_mean.map(lambda x : abs(x[0]-x[1])).coalesce(1)).repartition(100).map(lambda x :  (x[0], Vectors.dense(x[1])))
    two_sum_cosine = labels.zip(two_sum.map(lambda x : get_cosine(x)).coalesce(1)).repartition(100).map(lambda x : (x[0], Vectors.dense(x[1])))
    two_sum_sqeuclidean = labels.zip(two_sum.map(lambda x : sqeuclidean(x[0], x[1])).coalesce(1)).repartition(100).map(lambda x :  (x[0], Vectors.dense(x[1])))
    two_mean_cosine = labels.zip(two_mean.map(lambda x : get_cosine(x)).coalesce(1)).repartition(100).map(lambda x :  (x[0], Vectors.dense(x[1])))
    two_mean_sqeuclidean = labels.zip(two_mean.map(lambda x : sqeuclidean(x[0], x[1])).coalesce(1)).repartition(100).map(lambda x :  (x[0], Vectors.dense(x[1])))
    three = labels.zip(three.coalesce(1)).map(lambda x : (x[0], Vectors.dense(x[1])))
    four_sum_cosine = labels.zip(four_sum.map(lambda x : get_cosine(x)).coalesce(1)).repartition(100).map(lambda x :  (x[0], Vectors.dense(x[1])))
    four_sum_sqeuclidean = labels.zip(four_sum.map(lambda x : sqeuclidean(x[0], x[1])).coalesce(1)).repartition(100).map(lambda x :  (x[0], Vectors.dense(x[1])))
    four_mean_cosine = labels.zip(four_mean.map(lambda x : get_cosine(x)).coalesce(1)).repartition(100).map(lambda x :  (x[0], Vectors.dense(x[1])))
    four_mean_sqeuclidean = labels.zip(four_mean.map(lambda x : sqeuclidean(x[0], x[1])).coalesce(1)).repartition(100).map(lambda x : (x[0], Vectors.dense(x[1])))
    five_sum_cosine = labels.zip(five_sum.map(lambda x : (get_cosine(x[0]), x[1])).coalesce(1)).map(lambda x : (x[0], Vectors.dense(x[1][0], x[1][1])))
    five_sum_sqeuclidean = labels.zip(five_sum.map(lambda x : (sqeuclidean(x[0][0], x[0][1]), x[1])).coalesce(1)).map(lambda x : (x[0], Vectors.dense(x[1][0], x[1][1])))
    five_mean_cosine = labels.zip(five_mean.map(lambda x : (get_cosine(x[0]), x[1])).coalesce(1)).map(lambda x : (x[0], Vectors.dense(x[1][0], x[1][1])))
    five_mean_sqeuclidean = labels.zip(five_mean.map(lambda x : (sqeuclidean(x[0][0], x[0][1]), x[1])).coalesce(1)).map(lambda x : (x[0], Vectors.dense(x[1][0], x[1][1])))
    RDDs = [one_sum_difference, one_mean_difference, two_sum_cosine, two_sum_sqeuclidean,
            two_mean_cosine, two_mean_sqeuclidean, three, four_sum_cosine, four_sum_sqeuclidean,
            four_mean_cosine, four_mean_sqeuclidean, five_sum_cosine, five_mean_sqeuclidean,
            five_mean_cosine, five_mean_sqeuclidean]
    return RDDs





train = sc.textFile("hdfs:///shared/quora/train.csv", minPartitions=100)
train = train.map(lambda x : x.replace('\n' , '')).mapPartitions(lambda x : csv.reader(x)).filter(lambda x : x[0] != 'id' ).filter(lambda x : len(x) == 6  )
train_questions = train.map(lambda x : (clean_sentence(x[3]),clean_sentence( x[4])))
train_labels = train.map(lambda x : int(x[5]))
train_corpus = train_questions.map(lambda x : x[0]).collect() + train_questions.map(lambda x : x[1]).collect()
train_weights = tfidf(train_corpus)

train = get_RDDs(train, train_corpus, train_weights, train_questions, train_labels)
answers = get_results(train)

answers.coalesce(1).saveAsTextFile('hdfs:///user/osmarjosh/net')
