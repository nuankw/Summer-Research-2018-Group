'''
Chen Song, Nuan Wen, Yunkai Zhang 08/17/2018
Create csv files for common covariants such as age, day of the week, etc, each being a single row.
'''
import csv
import numpy as np

filename = './data/electricity_hourly.csv'
with open(filename) as f:
	reader = csv.reader(f,delimiter=',')
	data = np.array(list(reader))
	data = data[1: , 1:]

## age ##
age_single = [i for i in range(data.shape[0])]
with open("./data/age.csv","w") as f:
	writer = csv.writer(f)
	writer.writerow(age_single)

### day of the week ###
day_of_the_week_single=[]
for i in range(7):
	day_of_the_week_single += [i+1 for j in range(24)]
day_of_the_week=[]
for i in range(data.shape[0]//(7*24)+1):
	day_of_the_week += day_of_the_week_single
day_of_the_week = day_of_the_week[0:data.shape[0]]
with open("./data/day_of_the_week.csv","w") as f:
	writer = csv.writer(f)
	writer.writerow(day_of_the_week)

### hour of the day ###
hour_of_the_day_single=[i+1 for i in range(24)]
hour_of_the_day = []
for i in range(data.shape[0]//(24)+1):
	hour_of_the_day+=hour_of_the_day_single
hour_of_the_day=hour_of_the_day[0:data.shape[0]]
with open("./data/hour_of_the_day.csv","w") as f:
	writer = csv.writer(f)
	writer.writerow(hour_of_the_day)
