from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import csv
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


ps = PorterStemmer()

categories = ["neutral","anger", "disgust", "fear", "Guilt", "Interest", "Joy", "Sad", "shame", "surprise"]
stopwords = stopwords.words("english")

count  = 0
text = []
cat =[]
#
# text1 = []
# cat1= []
#
# text2 = []
# cat2 =[]
#
# text3 = []
# cat3 =[]
#
# text4 = []
# cat4 =[]
# text5 = []
# cat5 =[]
# text6 = []
# cat6 =[]
# text7 = []
# cat7 =[]
# text8 = []
# cat8 =[]
# text9 = []
# cat9 =[]
#

with open("tweets.csv") as tweet:
    reader = csv.reader(tweet)
    for row in reader:
        text.append(row[0])
        cat.append(row[1])


# with open("twitterDataSad.csv") as sad:
#     reader = csv.reader(sad)
#     for row in reader:
#         text1.append(row[0])
#         cat1.append(row[1])
#         count += 1
#         if count >= 445:
#             break
#
# count = 0
# with open("twitterDataJoy.csv") as joy:
#     reader2 = csv.reader(joy)
#     for row in reader2:
#         text2.append(row[0])
#         cat2.append(row[1])
#         if count >= 445:
#             break
#
# count = 0
# with open("twitterDataAngry.csv") as anger:
#     reader3 = csv.reader(anger)
#     for row in reader3:
#         text3.append(row[0])
#         cat3.append(row[1])
#         if count >= 445:
#             break
#
# count = 0
# with open("twitterDataSurprise.csv") as surprise:
#     reader4 = csv.reader(surprise)
#     for row in reader4:
#         text4.append(row[0])
#         cat4.append(row[1])
#         if count >= 445:
#             break
#
# count = 0
# with open("twitterDataDisgust.csv") as disgust:
#     reader5 = csv.reader(disgust)
#     for row in reader5:
#         text5.append(row[0])
#         cat5.append(row[1])
#         if count >= 445:
#             break
#
#
# count = 0
# with open("twitterDataFear.csv") as fear:
#     reader6 = csv.reader(fear)
#     for row in reader6:
#         text6.append(row[0])
#         cat6.append(row[1])
#         if count >= 445:
#             break
#
#
# count = 0
# with open("twitterDataGuilty.csv") as guilty:
#     reader7 = csv.reader(guilty)
#     for row in reader7:
#         text7.append(row[0])
#         cat7.append(row[1])
#         if count >= 445:
#             break
#
# count = 0
# with open("twitterDataInterest.csv") as interest:
#     reader8 = csv.reader(interest)
#     for row in reader8:
#         text8.append(row[0])
#         cat8.append(row[1])
#         if count >= 445:
#             break
#
# count = 0
# with open("twitterDataShame.csv") as shame:
#     reader9 = csv.reader(shame)
#     for row in reader9:
#         text9.append(row[0])
#         cat9.append(row[1])
#         if count >= 445:
#             break
#
#
# for i in range(445):
#     text.append(text1[i])
#     cat.append(cat1[i])
#     text.append(text2[i])
#     cat.append(cat2[i])
#     text.append(text3[i])
#     cat.append(cat3[i])
#     text.append(text4[i])
#     cat.append(cat4[i])
#     text.append(text5[i])
#     cat.append(cat5[i])
#     text.append(text6[i])
#     cat.append(cat6[i])
#     text.append(text7[i])
#     cat.append(cat7[i])
#     text.append(text8[i])
#     cat.append(cat8[i])
#     text.append(text9[i])
#     cat.append(cat9[i])


#Preprocessing task
temp = []
for i in range(4000):
    temp=word_tokenize(text[i], language="english")
    #print(temp)
    text[i] = ""
    for j in range(len(temp)):
        if(temp[j] not in stopwords):
            temp[j] = ps.stem(temp[j])
            text[i] += temp[j] + " "



import csv
sadFile = open("tweets.csv", "w")
writer = csv.writer(sadFile, delimiter=',')

with open("twitterDataSad.csv") as sad:
    for i in range (4003):
        row = text[i] + "," + cat[i] + "\n"
        sadFile.write(row)
sadFile.close()

trainText = text[:800]
trainCat = cat[:800]
testText = text[800:]
testCat = cat[800:]

#Bag-of-Words
#tokenizing using scikit-learn
#count_vect = CountVectorizer()
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trainText)

#Finding TF_IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf)

#Traing classifiers

#Naive-bayes classifers
classifier = MultinomialNB().fit(X_train_tfidf, trainCat)

# #Building the pipeline

from sklearn.pipeline import Pipeline
text_classifier = Pipeline([('vectorizer', CountVectorizer()),('tfidfFinder', TfidfTransformer()),
                     ('classifier', MultinomialNB()),])

text_clf = text_classifier.fit(trainText, trainCat)

# #Evaluation
import numpy as np
doc_test = testText
predicted  = text_clf.predict(doc_test)
print("NAIVE BAYES ACCURACY",np.mean(predicted == testCat))


from sklearn.neural_network import MLPClassifier
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                     ('clf', MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000)),])
text_clf= text_clf.fit(trainText, trainCat)
predicted = text_clf.predict(doc_test)
print("NN ACCURACY: ",np.mean(predicted == testCat))

#using Support vector machine

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
text_clf= text_clf.fit(trainText, trainCat)
predicted = text_clf.predict(doc_test)
print("SVM ACCURACY: ",np.mean(predicted == testCat))


#Further tuning
from sklearn import metrics
print(metrics.classification_report(testCat, predicted, target_names=categories))
print(metrics.confusion_matrix(testCat, predicted))

