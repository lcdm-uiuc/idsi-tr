from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("Movie Review Classification")
sc = SparkContext(conf=conf)

from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.classification import NaiveBayes
import json
import nltk
import csv
from nltk.tokenize import RegexpTokenizer

'''Filters and cleans the movie data to get rid of stop words and return (score, text)'''
def map_correct_format(split_row):
    text = split_row[0]
    score = split_row[-1]
    if score.isdigit() and text:
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        stop = set(nltk.corpus.stopwords.words('english'))
        text = [word.lower() for word in text if word.lower() not in stop]
        score = float(score)
        return (score, text)
    return None

'''Map the data from the filtered set'''
def map_from_portable(split_row):
    score = split_row[0]
    if float(score) > 5:
        score = 1
    else:
        score = 0
    text = split_row[1]
    text = [word for word in text.split(' ')]
    return (score, text)

'''Filters the data for use with sci kit learn'''
def map_portable_format(split_row):
    text = split_row[0]
    score = split_row[-1]
    if score.isdigit() and text:
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        stop = set(nltk.corpus.stopwords.words('english'))
        text = [word.lower() for word in text if word.lower() not in stop]
        score = float(score)
        return str(score) + "\t" + " ".join(text)
    return None

# This code was used just to clean the data and remove stop words
# reviews = sc.textFile("hdfs:///user/davidw1339/SAR14.txt")
# reviews = reviews.mapPartitions(lambda x: csv.reader(x, delimiter=','))
# words_score = reviews.map(map_portable_format)
# words_score = words_score.filter(lambda x: x)
# words_score.coalesce(1).saveAsTextFile('hdfs:///user/davidw1339/filtered_movie_reviews')

reviews = sc.textFile("hdfs:///user/davidw1339/filtered_movie_reviews/part-00000")
reviews = reviews.mapPartitions(lambda x: csv.reader(x, delimiter='\t'))
labeled_data = reviews.map(map_from_portable)

labels = labeled_data.map(lambda x: x[0])

# Feed HashingTF just the array of words
tf = HashingTF().transform(labeled_data.map(lambda x: x[1]))
# Pipe term frequencies into the IDF
idf = IDF(minDocFreq=50).fit(tf)
# Transform the IDF into a TF-IDF
tfidf = idf.transform(tf)

# Reassemble the data into (label, feature) K,V pairs
zipped_data = (labels.zip(tfidf)
                     .map(lambda x: LabeledPoint(x[0], x[1]))
                     .cache())
# Do a random split so we can test our model on non-trained data
training, test = zipped_data.randomSplit([0.7, 0.3])
# Train our model with the training data
model = NaiveBayes.train(training)

train_preds = (training.map(lambda x: x.label)
                       .zip(model.predict(training.map(lambda x: x.features))))
# Use the test data and get predicted labels from our model
test_preds = (test.map(lambda x: x.label)
                  .zip(model.predict(test.map(lambda x: x.features))))

# Evaluate the model using MulticlassMetrics
trained_metrics = MulticlassMetrics(train_preds.map(lambda x: (x[0], float(x[1]))))
test_metrics = MulticlassMetrics(test_preds.map(lambda x: (x[0], float(x[1]))))

print(trained_metrics.confusionMatrix().toArray())
print(trained_metrics.precision())

print(test_metrics.confusionMatrix().toArray())
print(test_metrics.precision())

metrics = [trained_metrics.confusionMatrix().toArray(), trained_metrics.precision(), test_metrics.confusionMatrix().toArray(), test_metrics.precision()]
sc.parallelize(metrics).coalesce(1).saveAsTextFile('hdfs:///user/davidw1339/movie_review_classification_mllib')
