from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("Stock Linear Regression Average")
sc = SparkContext(conf=conf)

import csv

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.regression import LinearRegressionWithSGD

'''Maps data from csv to a (stock,time) point'''
def map_to_point(row):
    date = row[2]
    time = row[3]
    value = row[4]
    try:
        date = float(date)
        value = float(value)
        #calculate time offset to add to date e.g. 18422.5 (if halway through day)
        h, m, s = time.split(':')
        offset = (int(h) * 60 + int(m))/(float(1440))
        date += offset
        time_vector = []
        time_vector.append(round(date, 1))
        return (value, time_vector)
    except ValueError:
        return None

def map_percent_error(data_point):
    # data_point consists of (y_true, y_pred)
    # formula: (abs((y_true - y_pred) / y_true)) * 100
    y_true = data_point[0]
    y_pred = data_point[1]
    if y_true != 0:
        return (abs((y_true - y_pred) / y_true) * 100, 1)
    return None

# change this parameter to vary how much data the script will be run on
file_size = '1000'

# read in the list of acceptable files based on the datasize we want
selected_files = []
files = sc.textFile('hdfs:///user/davidw1339/stockdatafiles' + file_size + '.txt')
selected_files = files.collect()

# list that stores the metrics for each parsed file
metrics = []
for selected_file in selected_files:
    # get the data from each stock csv file
    stocks = sc.textFile("hdfs:///shared/financial_data/stocks/permno_csv/" + selected_file)
    stocks = stocks.mapPartitions(lambda x: csv.reader(x))
    # map and filter the data to (stock, time)
    labeled_data = stocks.map(map_to_point)
    labeled_data = labeled_data.filter(lambda x: x)
    labeled_data = labeled_data.map(lambda x: LabeledPoint(x[0], x[1])).cache()
    training, test = labeled_data.randomSplit([0.7, 0.3])
    # verify that the data exists
    if training.isEmpty():
        metrics.append([])
        continue
    # train the model
    model = LinearRegressionWithSGD.train(training, iterations=1000, step=0.00000001, intercept=True)
    test_features = test.map(lambda x: x.features)
    predictions = model.predict(test_features)
    test_preds = test.map(lambda x: x.label).zip(predictions)

    # grab percent error
    total_percent = test_preds.map(map_percent_error)
    total_percent = total_percent.filter(lambda x: x)
    # check to make sure not empty rdd
    if total_percent.isEmpty():
        metrics.append([])
        continue
    average_percent = total_percent.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    average_percent = average_percent[0] / average_percent[1]

    # using regression metrics to get additional output values to judge model
    test_metrics = RegressionMetrics(test_preds.map(lambda x: (x[0], float(x[1]))))
    metrics.append([test_metrics.explainedVariance, test_metrics.meanSquaredError, average_percent])

# write all the data to a file
sc.parallelize(metrics).coalesce(1).saveAsTextFile('hdfs:///user/davidw1339/stock_linear_regression' + file_size)
