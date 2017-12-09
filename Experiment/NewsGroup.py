from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

#print(twenty_train.data[0])


print("Type of twenty train: ", type(twenty_train.data))
print(twenty_train.target_names)
print(len(twenty_train.data))
print(twenty_train.target[:10])

# for t in twenty_train.target[:10]:
#     print(twenty_train.target_names[t])

#Bag-of-Words
#tokenizing using scikit-learn

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'algorithm'))

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
classifier = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

#finding class of unkonow doc

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = classifier.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

#Building the pipeline
from sklearn.pipeline import Pipeline
text_classifier = Pipeline([('vectorizer', CountVectorizer()),('tfidfFinder', TfidfTransformer()),
                     ('classifier', MultinomialNB()),])
text_clf = text_classifier.fit(twenty_train.data, twenty_train.target)

#Evaluation
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
                                 categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data

#print("Test Document: ",docs_test[1])
predicted = text_clf.predict(docs_test)
print("NAIVE BAYES ACCURACY",np.mean(predicted == twenty_test.target))

#using Support vector machine

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
text_clf= text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print("SVM ACCURACY: ",np.mean(predicted == twenty_test.target))

#Further tuning

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
metrics.confusion_matrix(twenty_test.target, predicted)


