'''This file was just used to read output from the mllibregression script
and print out average variance, mean squared error, and percent error values'''

file_name = "stock_mllib_15000_avg.txt"

# collection of variables that track totals for each metric
total_variance = 0
total_mse = 0
total_percent = 0
num = 0

with open(file_name, 'r') as stock_file:
    for line in stock_file:
        # reformat and filter the line of data
        line.strip()
        line = line[1:-2]
        if len(line) == 0:
            continue
        # grab the corresponding values from the filtered line
        variance, mse, percent_error = line.split(',')
        # cast and add the values to the running totals
        variance = float(variance.strip())
        mse = float(mse.strip())
        percent_error = float(percent_error.strip())
        total_variance += variance
        total_mse += mse
        total_percent += percent_error
        num += 1

# print out the final averages
print(total_variance/num, total_mse/num, total_percent/num)
