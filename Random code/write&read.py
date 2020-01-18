import numpy as np
import csv

table = np.identity(5)

#write file
with open('test_file.csv','w') as csvfile:
	writer = csv.writer(csvfile)
	[writer.writerow(r) for r in table]

table = np.zeros((5,5))

#read it
with open('test_file.csv','r') as csvfile:
	reader = csv.reader(csvfile)
	table = [[float(e) for e in r] for r in reader]

print(table)