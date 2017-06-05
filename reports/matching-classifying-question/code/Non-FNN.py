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

'''

This code was for our CS199 technical paper
"Matching and Judging Question Similarity Using Rudimentary Techniques".
To run this code, you need to set up Pyspark and import all of the necessary
libraries to run the code.

@paper : {link}

@authors : Joshua Dunigan, Osmar Coronel, Professor Robert J. Brunner
'''

conf = SparkConf().setAppName("project")
sc = SparkContext(conf=conf)


'''
Cleans each sentence by removing punctuation, lowercasing all letters,
and removing special characters that may cause the text vectorizing methods
errors.

@input : line - A string

@returns : cleanedString - A string
'''
def clean_sentence(line):
    cleanedString = ""
    line = re.sub(r'[^\w\s]','',line)
    for letter in line:
        if ord(letter) > 127:
            cleanedString += str(ord(letter))
        else:
            cleanedString += letter.lower()
    return cleanedString

'''
Takes in an array of words and returns an array of tfidf scores
for each word in the array

@input : weights - Dictionary of tfidf weights.
         sentence - A list of strings.

@returns : vector - A list of tfidf scores for each word.
'''
def weight_vector_tfidf(weights, sentence):
    vector = []
    for word in sentence:
        if word is not '':
            tfidfScore = weights.get(word)
            vector.append(tfidfScore)
    return vector

'''
Takes in an array of words and returns an array of word2vec
vectors for each given word.

@input : w2v - the word2vec model containing the vectors for each word
         sentence - A list of strings
'''
def weight_vector_w2v(w2v, sentence):
    vector = []
    for word in sentence:
        if word is not '':
            w2vVector = w2v.wv[word]
            vector.append(w2vVector)
    return vector

'''
Takes in an array of words and returns an array of word2vec
vectors weighted by the words tfidf score.

@input : weights - Dictionary of tfidf weights
         w2v - the word2vec model containing the vectors for each word
         sentence - A list of strings

@returns : vector - A list of word2vec vectors weighted by tfidf scores
'''
def weight_vector_both(weights, w2v, sentence):
    vector = []
    for word in sentence:
        if word is not '':
            w2vVector = w2v.wv[word]
            tfidfScore = weights.get(word)
            vector.append(w2vVector*tfidfScore)
    return vector

'''
Gets the tfidf score for each word.

@input : eps - A smoothing constant
         count - The total occurances of the word in the corpus

@returns : the tfidf weight for the current word.
'''
def get_weight(count, eps=10000, min_count=1):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

'''
This code contains where we found the code to calculate the tfidf score for words.
We used this implementation because it was easier for combining the tfidf and
word2vec scores.

The method is a basic determinator of a tfidf score

@input : Corpus - A list of lists of strings, the lists of strings are the cleaned sentences

@return : weights - A dictionary containing the tfidf score for each unique word
'''
def tfidf(corpus):
    eps = 5000
    words = [x for sublist in corpus for x in sublist]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    return weights

'''
Calculates the Jaccard Index for two sentences, which is calculated as
the size of the set intersection over the sum of the size of the two sets
minus the size of the set union.

 |A interesction with B |
___________________________
|A| + |B| - |A union with B|

@input : sentenceA - A list of strings
         sentenceB - A list of strings

@returns : The jaccard index for two sentences
'''
def jaccard_index(sentenceA, sentenceB):
    sentenceA = set(sentenceA)
    sentenceB = set(sentenceB)
    setIntersection = sentenceA.intersection(sentenceB)
    return float(len(setIntersection)) / (len(sentenceA) + len(sentenceB) - len(setIntersection))

'''
Calculates the sum of the tfidf scores in the array.

@input : array - A list of floats (tfidf score for each word)

@return : total - The sum of all the tfidf scores in the sentence
'''
def get_sum(array):
    total = 0
    for vector in array:
        total += vector
    return total

'''
Calculates the mean of the tfidf scores in the array.

@input : array - A list of floats (tfidf score for each word)

@return : total - The mean of all the tfidf scores in the sentence
'''
def get_mean(array):
    total = 0
    for vector in array:
        total += vector
    if len(array) != 0:
        return total / len(array)
    return total

'''
Sums the word2vec vectors into one vector.

@input : sentence - A list of the numpy vectors (the word2vec vectors for each word)

@return : newVector - The sum of all the word2vec vectors
'''
def sum_w2v(sentence):
    newVector = [0]*100
    for vector in sentence:
        for i in range(len(vector)):
            newVector[i] += vector[i]
    newVector = np.array(newVector)
    return newVector

'''
Gets the mean of all the word2vec vectors.

@input : sentence - A list of the numpy vectors (the word2vec vectors for each word)

@return : newVector - The mean of all the word2vec vectors
'''
def mean_w2v(sentence):
    newVector = [0]*100
    for vector in sentence:
        for i in range(len(vector)):
            newVector[i] += vector[i]
    newVector = np.array(newVector)
    return newVector / len(sentence)

'''
Calulates the cosine distance of two vectors. Returns 1.0 if
there is an error in calculating, meaning they are not similar.

@input : pair_of_vectors - A tuple of numpy vectors (the word2vec vectors summed/averaged)

@returns : cosineScore - The cosine distance between two vectors
'''
def get_cosine(pair_of_vectors):
    try:
        cosineScore = cosine(pair_of_vectors[0],pair_of_vectors[1])
    except Exception as e:
        cosineScore = 1.0
    return cosineScore

'''
Reads in the list of RDDs to predict on. Outputs an RDD to save as a text file.

Models can be swapped out for either of the models listed below.

model = NaiveBayes.train(training)
model = LogisticRegressionWithSGD.train(training, iterations=10)

The neural net does not work for Pyspark 1.5.2, so you need to use the code listed in the neuralnet.py file.

@input : train - A list of RDDs from train.csv that contained the data to train and predict on
         test - A list of RDDs from test.csv that contained the data to train and predict on

@returns : The precision and confusion matrix for each of the RDDs the model predicted
'''
def get_results(train, test):
    results = []
    for i in range(len(train)):
        training = train[i].filter(lambda x : not np.isnan(x.features[0])).filter(lambda x : x.features[0] > 0.0 )
        testing = test[i].filter(lambda x : not np.isnan(x.features[0])).filter(lambda x : x.features[0] > 0.0 )
        model = RandomForest.trainClassifier(training, numClasses=2, categoricalFeaturesInfo={},numTrees=20, featureSubsetStrategy="auto",impurity='gini', maxDepth=10, maxBins=32)
        test_preds = (testing.map(lambda x: x.label).zip(model.predict(testing.map(lambda x: x.features))))
        test_metrics = MulticlassMetrics(test_preds.map(lambda x: (x[0], float(x[1]))))
        answer = str(test_metrics.precision()) + '\n' + str(test_metrics.confusionMatrix().toArray()) + '\n'
        results.append(answer)
    return sc.parallelize(results)

'''
Creates all of the RDDs with the difference ways of getting features in the training
and test data

@input : data - The original RDD
         corpus - The list of sentence lists
         weights - The dictionary of all the tfidf scores for the unique words
         questions - The RDD containing the pairs of questions

@returns data - The list of RDDs containing the different features for the model to predict
'''
def get_RDDs(data, corpus, weights, questions, labels):
    w2v = word2vec.Word2Vec(corpus, size=100, window=20, min_count=1, workers=40)

    '''
    one : tfidf scores only
    two : word2vec vectors only
    three : jaccard index only
    four : word2vec * tfidf
    five : word2vec * tfidf, jaccard index

    sum or mean : way the word vectors for the entire sentence
                  were combined into one vector or number

    cosine or squeclidean : similarity measurement on the two sum/mean vectors
    '''
    one = questions.map(lambda x : (weight_vector_tfidf(weights, x[0]), weight_vector_tfidf(weights, x[1])))
    one_sum = one.map(lambda x : (get_sum(x[0]), get_sum(x[1]) ))
    one_mean = one.map(lambda x : (get_mean(x[0]), get_mean(x[1])))
    two = questions.map(lambda x : (weight_vector_w2v(w2v, x[0]), weight_vector_w2v(w2v, x[1])))
    two_sum = two.map(lambda x : (sum_w2v(x[0]), sum_w2v(x[1])))
    two_mean = two.map(lambda x : (mean_w2v(x[0]), mean_w2v(x[1])))
    three = questions.map(lambda x : jaccard_index(x[0], x[1]))
    four = questions.map(lambda x : (weight_vector_both(weigts, w2v, x[0]), weight_vector_both(weights, w2v, x[1])))
    four_sum = four.map(lambda x : (sum_w2v(x[0]), sum_w2v(x[1])))
    four_mean = four.map(lambda x : (mean_w2v(x[0]), mean_w2v(x[1])))
    five_sum = four_sum.zip(three)
    five_mean = four_mean.zip(three)
    labels = labels.coalesce(1)
    one_sum_difference = labels.zip(one_sum.map(lambda x : abs(x[0]-x[1])).coalesce(1)).repartition(100).map(lambda x : LabeledPoint(x[0], [x[1]]))
    one_mean_difference  = labels.zip(one_mean.map(lambda x : abs(x[0]-x[1])).coalesce(1)).repartition(100).map(lambda x : LabeledPoint(x[0], [x[1]]))
    two_sum_cosine = labels.zip(two_sum.map(lambda x : get_cosine(x)).coalesce(1)).repartition(100).map(lambda x : LabeledPoint(x[0], [x[1]]))
    two_sum_sqeuclidean = labels.zip(two_sum.map(lambda x : sqeuclidean(x[0], x[1])).coalesce(1)).repartition(100).map(lambda x : LabeledPoint(x[0], [x[1]]))
    two_mean_cosine = labels.zip(two_mean.map(lambda x : get_cosine(x)).coalesce(1)).repartition(100).map(lambda x : LabeledPoint(x[0], [x[1]]))
    two_mean_sqeuclidean = labels.zip(two_mean.map(lambda x : sqeuclidean(x[0], x[1])).coalesce(1)).repartition(100).map(lambda x : LabeledPoint(x[0], [x[1]]))
    three = labels.zip(three.coalesce(1)).map(lambda x : LabeledPoint(x[0], [x[1]]))
    four_sum_cosine = labels.zip(four_sum.map(lambda x : get_cosine(x)).coalesce(1)).repartition(100).map(lambda x : LabeledPoint(x[0], [x[1]]))
    four_sum_sqeuclidean = labels.zip(four_sum.map(lambda x : sqeuclidean(x[0], x[1])).coalesce(1)).repartition(100).map(lambda x : LabeledPoint(x[0], [x[1]]))
    four_mean_cosine = labels.zip(four_mean.map(lambda x : get_cosine(x)).coalesce(1)).repartition(100).map(lambda x : LabeledPoint(x[0], [x[1]]))
    four_mean_sqeuclidean = labels.zip(four_mean.map(lambda x : sqeuclidean(x[0], x[1])).coalesce(1)).repartition(100).map(lambda x : LabeledPoint(x[0], [x[1]]))
    five_sum_cosine = labels.zip(five_sum.map(lambda x : (get_cosine(x[0]), x[1])).coalesce(1)).map(lambda x : LabeledPoint(x[0], [x[1][0], x[1][1]]))
    five_sum_sqeuclidean = labels.zip(five_sum.map(lambda x : (sqeuclidean(x[0][0], x[0][1]), x[1])).coalesce(1)).map(lambda x : LabeledPoint(x[0], [x[1][0], x[1][1]]))
    five_mean_cosine = labels.zip(five_mean.map(lambda x : (get_cosine(x[0]), x[1])).coalesce(1)).map(lambda x : LabeledPoint(x[0], [x[1][0], x[1][1]]))
    five_mean_sqeuclidean = labels.zip(five_mean.map(lambda x : (sqeuclidean(x[0][0], x[0][1]), x[1])).coalesce(1)).map(lambda x : LabeledPoint(x[0], [x[1][0], x[1][1]]))
    RDDs = [one_sum_difference, one_mean_difference, two_sum_cosine, two_sum_sqeuclidean,
            two_mean_cosine, two_mean_sqeuclidean, three, four_sum_cosine, four_sum_sqeuclidean,
            four_mean_cosine, four_mean_sqeuclidean, five_sum_cosine, five_mean_sqeuclidean,
            five_mean_cosine, five_mean_sqeuclidean]

    return RDDs

# Load in the csv file into an RDD
train = sc.textFile("hdfs:///shared/quora_kaggle/train.csv", minPartitions=100)
train = train.map(lambda x : x.replace('\n' , '')).mapPartitions(lambda x : csv.reader(x)).filter(lambda x : x[0] != 'id' ).filter(lambda x : len(x) == 6  )

# Maps the labels, question pairs, and each question to RDDs
train_questions = train.map(lambda x : (clean_sentence(x[3]),clean_sentence( x[4])))
train_labels = train.map(lambda x : int(x[5]))
train_corpus = train_questions.map(lambda x : x[0]).collect() + train_questions.map(lambda x : x[1]).collect()
train_weights = tfidf(train_corpus)

test = sc.textFile("hdfs:///shared/quora_kaggle/test.csv", minPartitions=100)
test = test.map(lambda x : x.replace('\n' , '')).mapPartitions(lambda x : csv.reader(x)).filter(lambda x : x[0] != 'test_id' ).filter(lambda x : len(x) == 3  )

# Maps the labels, question pairs, and each question to RDDs
test_questions = test.map(lambda x : (clean_sentence(x[1]),clean_sentence(x[2])))
test_corpus = test_questions.map(lambda x : x[0]).collect() + test_questions.map(lambda x : x[1]).collect()
test_weights = tfidf(test_corpus)

train = get_RDDs(train, train_corpus, train_weights, train_questions, train_labels)
test = get_RDDs(test, test_corpus, test_weights, test_questions, train_labels)

answers = get_results(train,test)

#Path to save the answers to
answers.coalesce(1).saveAsTextFile('hdfs:///user/joshuad2/Final')
