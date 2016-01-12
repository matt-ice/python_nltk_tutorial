from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats",pos="n")) #default pos = noun
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))

print(lemmatizer.lemmatize("better",pos="a"))
print(lemmatizer.lemmatize("best",pos="a"))
