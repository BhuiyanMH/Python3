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
#
# # #Preprocessing task
# temp = []
# for i in range(1600):
#     temp=word_tokenize(text[i], language="english")
#     #print(temp)
#     text[i] = ""
#     for j in range(len(temp)):
#         if(temp[j] not in stopwords):
#             temp[j] = ps.stem(temp[j])
#             text[i] += temp[j] + " "

trainText = text[1200:]
testText = text[:1200]
trainCat = cat[1200:]
testCat = cat[:1200]

#Bag-of-Words
#tokenizing using scikit-learn

count_vect = CountVectorizer()
count_vect = CountVectorizer(ngram_range=(1, 2), min_df=1)
X_train_counts = count_vect.fit_transform(trainText)

#Finding TF_IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# #print(X_train_tfidf)


# count_vect = CountVectorizer(ngram_range=(1, 2))
# text_counts = count_vect.fit_transform(text)
#
# #Finding TF_IDF
# tfidf_transformer = TfidfTransformer()
# text_tfidf = tfidf_transformer.fit_transform(text_counts)
#print(X_train_tfidf)

# X_train_tfidf = text_tfidf[1200:]
# testText = text_tfidf[:1200]

#Traing classifiers

#Naive-bayes classifers
#classifier = MultinomialNB().fit(X_train_tfidf, trainCat)

# #Building the pipeline

from sklearn.pipeline import Pipeline
text_classifier = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1, 2), min_df=1)),('tfidfFinder', TfidfTransformer()),
                     ('classifier', MultinomialNB()),])

text_clf = text_classifier.fit(trainText, trainCat)

# #Evaluation
import numpy as np
doc_test = testText
predicted  = text_clf.predict(doc_test)
print("NAIVE BAYES ACCURACY",np.mean(predicted == testCat))


from sklearn.neural_network import MLPClassifier
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), min_df=1)), ('tfidf', TfidfTransformer()),
                     ('clf', MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000)),])
text_clf= text_clf.fit(trainText, trainCat)
predicted = text_clf.predict(doc_test)
print("NN ACCURACY: ",np.mean(predicted == testCat))

#using Support vector machine

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), min_df=1)), ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
text_clf= text_clf.fit(trainText, trainCat)
predicted = text_clf.predict(doc_test)
print("SVM ACCURACY: ",np.mean(predicted == testCat))


#Further tuning
from sklearn import metrics
print(metrics.classification_report(testCat, predicted, target_names=categories))
metrics.confusion_matrix(testCat, predicted)


#
# #Generating ROC Curve
#
# import numpy as np
# from scipy import interp
# import matplotlib.pyplot as plt
# from itertools import cycle
#
# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import StratifiedKFold
#
# cv = StratifiedKFold(n_splits=6)
# mean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)
#
# colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', "red", "green", "black", "magenta"])
# lw = 2
#
# i = 0
# for (train, test), color in zip(cv.split(trainText, trainCat), colors):
#     probas_ = text_clf.fit(trainText[train], trainCat[train]).predict_proba(testText[test])
#     # Compute ROC curve and area the curve
#     fpr, tpr, thresholds = roc_curve(trainCat[test], probas_[:, 1])
#     mean_tpr += interp(mean_fpr, fpr, tpr)
#     mean_tpr[0] = 0.0
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, lw=lw, color=color,
#              label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
#
#     i += 1
# plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#          label='Luck')
#
# mean_tpr /= cv.get_n_splits(trainText, trainCat)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#          label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
#
#
# FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
# FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
# TP = np.diag(confusion_matrix)
#
# #TN = confusion_matrix.values.sum() - (FP + FN + TP)
#
# TN = (400)-
#
# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP)
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # False discovery rate
# FDR = FP/(TP+FP)
#
# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)
