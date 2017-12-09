from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import csv

text = []
cat =[]
count  = 0

text1 = []
cat1= []

text2 = []
cat2 =[]

text3 = []
cat3 =[]

text4 = []
cat4 =[]


with open("twitterDataSad.csv") as sad:
    reader = csv.reader(sad)
    for row in reader:
        text1.append(row[0])
        cat1.append(row[1])
        count += 1
        if count >= 400:
            break

count = 0
with open("twitterDataJoy.csv") as joy:
    reader2 = csv.reader(joy)
    for row in reader2:
        text2.append(row[0])
        cat2.append(row[1])
        if count >= 400:
            break

count = 0
with open("twitterDataAngry.csv") as anger:
    reader3 = csv.reader(anger)
    for row in reader3:
        text3.append(row[0])
        cat3.append(row[1])
        if count >= 400:
            break

count = 0
with open("twitterDataSurprise.csv") as surprise:
    reader4 = csv.reader(surprise)
    for row in reader4:
        text4.append(row[0])
        cat4.append(row[1])
        if count >= 400:
            break

for i in range(400):
    text.append(text1[i])
    cat.append(cat1[i])
    text.append(text2[i])
    cat.append(cat2[i])
    text.append(text3[i])
    cat.append(cat3[i])
    text.append(text4[i])
    cat.append(cat4[i])

trainText = text[1200:]
testText = text[:1200]
trainCat = cat[1200:]
testCat = cat[:1200]

# for i in range(10):
#     print(trainText[i])
#     print(trainCat[i])
#     print(testText[i])
#     print(testCat[i])


# categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
# twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

#print(twenty_train.data[0])

# print("Type of twenty train: ", type(twenty_train.data))
# print(twenty_train.target_names)
# print(len(twenty_train.data))
# print(twenty_train.target[:10])

# for t in twenty_train.target[:10]:
#     print(twenty_train.target_names[t])

#Bag-of-Words
#tokenizing using scikit-learn

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trainText)
print(X_train_counts.shape)

#print(count_vect.vocabulary_.get(u'algorithm'))

# #Finding TF
# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)
# print(X_train_tf.shape)

#Finding TF_IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

#Traing classifiers

#Naive-bayes classifers
classifier = MultinomialNB().fit(X_train_tfidf, trainCat)

#finding class of unkonow doc

# docs_new = ['God is love', 'OpenGL on the GPU is fast']
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#
# predicted = classifier.predict(X_new_tfidf)

# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, twenty_train.target_names[category]))


#Building the pipeline

from sklearn.pipeline import Pipeline
text_classifier = Pipeline([('vectorizer', CountVectorizer()),('tfidfFinder', TfidfTransformer()),
                     ('classifier', MultinomialNB()),])

text_clf = text_classifier.fit(trainText, trainCat)

#Evaluation
import numpy as np
# twenty_test = fetch_20newsgroups(subset='test',
#                                  categories=categories, shuffle=True, random_state=42)
# docs_test = twenty_test.data
# predicted = text_clf.predict(docs_test)
doc_test = testText
predicted  = text_clf.predict(doc_test)
print("NAIVE BAYES ACCURACY",np.mean(predicted == testCat))

#using Support vector machine

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
text_clf= text_clf.fit(trainText, trainCat)
predicted = text_clf.predict(doc_test)
print("SVM ACCURACY: ",np.mean(predicted == testCat))


from sklearn.neural_network import MLPClassifier
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                     ('clf', MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000)),])
text_clf= text_clf.fit(trainText, trainCat)
predicted = text_clf.predict(doc_test)
print("NN ACCURACY: ",np.mean(predicted == testCat))


#finding class of unkonow doc

docs_new = ['wow','Feeling joy']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = classifier.predict(X_new_tfidf)
print(predicted)


# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, twenty_train.target_names[category]))

# #Further tuning
#
# from sklearn import metrics
# print(metrics.classification_report(testCat, predicted,target_names=testCat))
# metrics.confusion_matrix(testCat, predicted)

