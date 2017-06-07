from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import time
import csv

time_labels = [] # set of stock times
price_labels = [] # the stock price corresponding to each time

'''Takes in a row of stock data and returns a mapped point (time, stock_price)'''
def parse_line(line):
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
        return (round(date, 1), value)
    except ValueError:
        return None

'''Formula to compute MAPE given output from model'''
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# the file size we are currently looking at in megabytes
file_size = '1000'

# read in the list of acceptable files based on the datasize we want
selected_files = []
with open('stockdatafiles' + file_size + '.txt', 'r') as f:
    for line in f:
        selected_files.append(line.strip())

# open a file to write output to
output_file = open('linregsklearn' + file_size + '.txt', 'w')

# start a timer
start = time.time()
num_files = len(selected_files)
for i in range(num_files):
    file_path = selected_files[i]
    time_labels = []
    price_labels = []
    # collect all the csv's that fall under our selected data set size
    with open('permno_csv/' + file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labeled_stock = parse_line(row)
            if(labeled_stock):
                time_labels.append([labeled_stock[0]])
                price_labels.append(labeled_stock[1])

    # split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(time_labels, price_labels, test_size=0.30, random_state=42)
    # filter out empty stock files
    if len(X_train) == 0 or len(y_train) == 0:
        continue
    # fit our linear regression model
    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # print out the error and time it took to train on each set of data
    error = str(mean_squared_error(y_test, y_pred))
    percent_error = str(mean_absolute_percentage_error(y_test, y_pred))
    d_time = str(i) + '/' + str(num_files) + ' Computed in: ' + str(time.time() - start)
    print(error)
    print(percent_error + '%')
    print(d_time)

    output_file.write(error + '\n')
    output_file.write(percent_error + '%\n')
    output_file.write(d_time + '\n')

output_file.close()
