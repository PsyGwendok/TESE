import csv
csvfile = open('images.csv','r')
csvFileArray = []
for row in csv.reader(csvfile, delimiter = ','):
    csvFileArray.append(row)

print(csvFileArray[0][0])