from pyspark import SparkContext, SparkConf
from gensim.models import word2vec
from numpy import array
from math import sqrt
from sklearn.neighbors import LSHForest
import csv
import re

conf = SparkConf().setAppName("Question Matching")
sc = SparkContext(conf=conf)

df = sc.textFile("hdfs:///user/joshuad2/a/part-00000", minPartitions=250)
data = df.mapPartitions(lambda x: csv.reader(x))


TAKEN = data.count()

'''
Clean the sentences in case of any unknown characters

line: str
	the line to be cleaned
'''
def clean_sentence(line):
    cleanStr = ""
    line = re.sub(r'[^\w\s]','',line)
    for letter in line:
        if ord(letter)>127:
            cleanStr += str(ord(letter))
        else:
            cleanStr += letter.lower()
    return cleanStr

#One data is for training and the other is for testing
cleanTrain = data.map(lambda x: clean_sentence(x[3]))
cleanTest = data.map(lambda x: clean_sentence(x[4]))

trainArray = cleanTrain.take(TAKEN)
testArray  = cleanTest.take(TAKEN)

dtNow = cleanTrain.map(lambda x: x.split())


#word2vec model
w2v = word2vec.Word2Vec(dtNow.take(TAKEN), size=3, window=20, min_count=1, workers=200)


'''
Turn the sentences into vectors by adding each word specific vector.

sentence : str
	the sentence to be turned into a vector
'''
def sentence2vec(sentence):
            sentenceVector = [0]*3
            for word in sentence:
                for num in range(len(sentenceVector)):
                    if word in w2v.wv.vocab:   
                        try:
                            sentenceVector[num] += tuple(w2v.wv[word])[num]
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

'''
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
parsedArray = parsed.take(TAKEN)

#Train model with the training dataset
lshf = LSHForest(random_state=42)
lshf.fit(parsedArray)

#Test model with the testing dataset
parsedTest = cleanTest.map(lambda x: x.split()).map(sentence2vec).map(cosineSimilarity)
parsedTestArray = parsedTest.take(TAKEN)

#write to csv file
with open('data.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(TAKEN):
        array = []
        distances, indices = lshf.kneighbors([parsedTestArray[i]], n_neighbors=1)
        array = [trainArray[indices[0][0]], testArray[i], 1 if distances[0][0]<.00005 else 0]
        writer.writerow(array)
        if distances[0][0] <.00005:
            print("distance: "+str(distances[0][0])+"\nQuestion1: "+trainArray[indices[0][0]]+"\nQuestion2: "+ testArray[i])