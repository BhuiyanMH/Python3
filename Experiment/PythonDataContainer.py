#Python List: Ordered sequence of heterogeneous items

list = [1, 2, 3, "Hello"]

print(list[0])
print(list[:1]) #upto 1, excluding
print(list[1:]) #from 1

#Python tuple: ordered sequence of list except, tuples are immutable

tuple = ("I", "wrote", 1, "python", "program")
print(tuple[1:])

#Python string: sequence of immutable Unicode characters, slicing operator
#is used to retreive the characters of string

para = "I am a student of CSE"
print(para[5:14])

#Python set:  is a unordered collection of  Unique items,
#as uniordered, no indexing, [] operator not works
#can perfoem union, intersection on two sets
soupSet = {"Bowl", "A", "Small Boul", "Spoons"}
print(soupSet)

#Python Dictionary: unordered collection of key-value pairs

wordDict = {"me":"ami", "you":"tumi", "he":"tini"}
print(wordDict["me"])