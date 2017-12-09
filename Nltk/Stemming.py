from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
ps = PorterStemmer()

example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

#for w in example_words:
#   print(ps.stem(w))

new_text = "It is importent to be pythonly when you are pythoning with python. All pythoners have pythoned poorly at least once"
words = word_tokenize(new_text)

print(words)
for w in words:
    print(ps.stem(w))