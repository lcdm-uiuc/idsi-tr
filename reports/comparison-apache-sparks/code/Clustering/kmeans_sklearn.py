from sklearn import cluster
import numpy as np
import time
import csv

start = time.time()

strings = []
coordinates =[]

with open("clusters.txt", 'r') as f:
    strings = f.read().split(",\n ")

for string in strings:
    string = string.replace("(", "").replace(")", "").replace(",", "").split()
    if len(string) == 2:
        a = float(string[0])
        b = float(string[1])
        coordinates.append((a,b))

k = 7
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(coordinates)

labels = kmeans.labels_
output = zip(coordinates, labels)

print('Computed in: ' + str(time.time() - start))
print(list(output))
