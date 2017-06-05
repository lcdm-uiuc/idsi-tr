from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
import numpy as np
import time
import csv

labels = [] # set of review scores
text = [] # the review text corresponding to each label

# parse a line of filtered data into (score, text) tuples
def parse_line(line):
    split_line = csv.reader(line, delimiter='\t')
    return(split_line[0], split_line[1])

start = time.time()

# open the file that contains results filtered from spark
with open('filtered_movie_reviews.txt') as f:
    movie_reader = csv.reader(f, delimiter='\t')
    for row in movie_reader:
        # append a 1 for a positive review and a 0 for a negative review
        if float(row[0]) > 5:
            labels.append(1)
        else:
            labels.append(0)
        text.append(row[1])

# run sk-learn's TFIDF vectorizer to extract the TF-IDF feature
vectorizer = TfidfVectorizer(min_df=5)
X = vectorizer.fit_transform(text)
# split the data for testing and training
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.30, random_state=42)

# fit the data to naive bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# print out the corresponding confusion matrix and precision score
print('Computed in: ' + str(time.time() - start))
print(confusion_matrix(y_test, y_pred, labels = [0, 1]))
print(precision_score(y_test, y_pred))
