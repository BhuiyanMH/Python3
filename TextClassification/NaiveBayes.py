import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# document = [list( movie_reviews.words(fileid)), category)
# for fileid in movie_reviews.fileids(categories)
#     for fileid in movie_reviews.fileids(category)]

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append( (list(movie_reviews.words(fileid)), category))

random.shuffle(documents)
#print(documents[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words["nice"])

word_features = list(all_words.keys())[:3000] #Top 3000 words
# #print(word_features)

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featurests = [(find_features(rev), category) for (rev, category) in documents]

training_set  = featurests[:1900]
testing_set = featurests[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)
#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Original Naive bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)

#classifier.show_most_informative_features(15)

# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

#Multinomial Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multinomial Naive bayes Algo accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

#Gassian Naive Bayes
# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print("Gaussian Naive bayes Algo accuracy: ", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

#Bernouli Naive Bayes
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("Bernouli Naive bayes Algo accuracy: ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

#LogisticRegression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Algo accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

#SGDClassifier
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Algo accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

#SVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC classification Algo accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

#LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Algo accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

#NuSVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Algo accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

