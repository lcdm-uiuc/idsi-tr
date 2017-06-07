# MatPlotlib
import matplotlib.pyplot as plt
from matplotlib import pylab

# Scientific libraries
import numpy as np
import scipy
from scipy.optimize import curve_fit
#plt.style.use('ggplot')

with open("../data/mllib_kmeans_artificial.txt", 'r') as f:
    strings = f.read().split("\n")

data = []
for string in strings:
    string = string.replace("(", "").replace(")", "").replace(",", "").replace("]", "").split()
    if len(string) == 3 and string[0] != "Computed" :
        a = float(string[0])
        b = float(string[1])
        c = int(string[2])
        data.append(((a,b),c))

coordinate, region = zip(*data)
x,y = zip(*coordinate)

#plt.style.use('ggplot')
plt.scatter(x,y, c = region)
plt.title('Clustering Artifically Clustered Data Using MLlib', fontsize=14)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)

pylab.savefig('../images/mllib_kmeans.png')
