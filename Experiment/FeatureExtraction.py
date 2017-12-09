from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
#print(vectorizer)
corpus = ['This is the first document.', 'This is the second second document.',
          'And the third one.', 'Is this the first document?',]
X = vectorizer.fit_transform(corpus)
#print(X)

print(vectorizer.get_feature_names())
arrayForm = X.toarray()
print(arrayForm)

#get the mapping: mapping is stored in "vocabulary_" attribute of the vectorizer
print("index of 'documnet': ",vectorizer.vocabulary_.get('document'))

newDataArray = vectorizer.transform(['Something completely new.']).toarray()
print(newDataArray)

#Bi-gram vectorization

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
print(analyze('Bi-grams are cool!'))
bigramArray = bigram_vectorizer.fit_transform(corpus).toarray()
print("Bigram array:\n", bigramArray)

#TF-IDF vectorization

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
print(transformer)

#Vectorizer and TFIDF transformer can be find together
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=1)
# vectorizer.fit_transform(corpus)

#Creating character N-Gram
# ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2), min_df=1)
# counts = ngram_vectorizer.fit_transform(['words', 'wprds'])
# print(ngram_vectorizer.get_feature_names())

#Hashing Vectorizer
# from sklearn.feature_extraction.text import HashingVectorizer
#     hv = HashingVectorizer(n_features=10)
#     hv.transform(corpus)