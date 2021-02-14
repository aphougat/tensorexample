import csv
import time
import numpy as np
import matplotlib.pyplot as plt


def read(filename, date_idx, date_parse, year, bucket=7):
    days_in_yr = 365
    freq = {}
    for period in range(0, int(days_in_yr/bucket)):
        freq[period] = 0
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row[date_idx] == '' or reader.line_num == 1:
                continue
            else:
                t = time.strptime(row[date_idx], date_parse)
                if t.tm_year == year and t.tm_yday < (days_in_yr - 1):
                    freq[int(t.tm_yday/bucket)] += 1
    return freq


freq = read("./data/311.csv", 1, '%m/%d/%Y %H:%M:%S %p', 2014)

x_train = np.asarray(list(freq.keys()))
y_train = np.asarray(list(freq.values()))
maxY = y_train.max()
ny_train = y_train / maxY
plt.scatter(x_train, ny_train)
#plt.show()

#print(freq)
