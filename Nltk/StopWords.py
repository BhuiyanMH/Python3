from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
example_sentence = "This is an example sentece showing off stopword filtration"
stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)

#filtered_sentence = []
#for w in words:
 #   if w not in words:
 #       filtered_sentence.append(w)

filtered_sentence = [w for w in words if not w in stopwords]
print(filtered_sentence)