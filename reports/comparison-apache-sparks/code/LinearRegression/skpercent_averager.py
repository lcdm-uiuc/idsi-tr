'''This file was just used to read output from the sklearnregression script
and print out average variance, mean squared error, and percent error values'''

file_name = "linregsklearn60000.txt"


# metrics holds [mean_squared_error, percent_error]
metrics = [0, 0]
num = 0

with open(file_name, 'r') as stock_file:
    counter = 0
    for line in stock_file:
        num += 1
        if counter == 0:
            metrics[counter] += float(line.strip())
        elif counter == 1:
            metrics[counter] += float(line.strip()[0:-1])
        else:
            counter = 0
            continue
        counter += 1

print(metrics[0]/num, metrics[1]/num)
