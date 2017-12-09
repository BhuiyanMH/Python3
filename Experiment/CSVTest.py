import csv
sadFile = open("tweetsSad.csv", "wb")
writer = csv.writer(sadFile, delimiter=',')

with open("twitterDataSad.csv") as sad:
    reader = csv.reader(sad)
    for row in reader:
        print(row[0])
        print(row[1])
        # for col in row:
        #     print(col)
        writer.writerow(row[0],row[1])
sadFile.close()
