import csv
import nltk
from sklearn.svm import LinearSVC

with open('C:/data/block.csv') as f:
    reader = csv.reader(f)
    next(reader)
    data = list(reader)

dataset = []
for row in data:
    # dataset.append(({'age': row[0], 'income': row[1], 'isStudent': row[2],
    #                  'creditRating': row[3]}, row[4]))

    dataset.append(({'Age': row[0], 'Sex': row[1], 'Hrate': row[2], 'Thalach': row[3], 'Cig': row[4], 'DM': row[5],
                     'HTN': row[6], 'BpSyst': row[7], 'BpDias': row[8], 'TrestbpsS': row[9], 'TrestbpsD': row[10],
                     'Cp': row[11], 'cough': row[12], 'Obs': row[13], 'Interpolation': row[14], 'Fib': row[15],
                     'AVNRT': row[16], 'Vtac': row[17], 'StSlopECG': row[18], 'Axis': row[19], 'Exang': row[20],
                     'TrestEcg': row[21], 'DCM': row[22], 'Thal': row[23]}, row[24]))
train_set = dataset[:150]
test_set = dataset[150:]

classifier = nltk.DecisionTreeClassifier.train(train_set, entropy_cutoff=0,support_cutoff=0)
print('Decision Tree Classification Accuracy: ', nltk.classify.accuracy(classifier, test_set)*100)

classifier = nltk.NaiveBayesClassifier.train(train_set)
print('Naive Bayes Classification Accuracy: ', nltk.classify.accuracy(classifier, test_set)*100)


classifier = nltk.classify.SklearnClassifier(LinearSVC())
classifier.train(train_set)
print('Support Vector Machine Classification Accuracy: ', nltk.classify.accuracy(classifier, test_set)*100)

