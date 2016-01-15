import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

documents =[]

for r in short_pos.split("\n"):
    documents.append((r,"pos"))
for r in short_neg.split("\n"):
    documents.append((r,"neg"))

all_words= []
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())
for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Original Naive Bayes Algorithm accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train((training_set))
print("MNB_classifier Naive Bayes Algorithm accuracy percent: ",
      (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train((training_set))
print("BernoulliNB_classifier Naive Bayes Algorithm accuracy percent: ",
      (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train((training_set))
print("LogisticRegression_classifier Naive Bayes Algorithm accuracy percent: ",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train((training_set))
print("SGDClassifier_classifier Naive Bayes Algorithm accuracy percent: ",
      (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train((training_set))
print("LinearSVC_classifier Naive Bayes Algorithm accuracy percent: ",
      (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train((training_set))
print("NuSVC_classifier Naive Bayes Algorithm accuracy percent: ",
      (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

voted_classifier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)
print("Voted_classifier Naive Bayes Algorithm accuracy percent: ",
      (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

#pickle
# pickle.dump(classifier, open("pos_neg_NB.p","wb"))
# pickle.dump( word_features, open('word_features.p','wb'))
# pickle.dump(featuresets, open('featuresets.p','wb'))
# pickle.dump(MNB_classifier, open('MNB_classifier.p','wb'))
# pickle.dump(BernoulliNB_classifier, open('BernoulliNB_classifier.p','wb'))
# pickle.dump(LogisticRegression_classifier, open('LogisticRegression_classifier.p','wb'))
# pickle.dump(SGDClassifier_classifier, open('SGDClassifier_classifier.p','wb'))
# pickle.dump(LinearSVC_classifier, open('LinearSVC_classifier.p','wb'))
# pickle.dump(NuSVC_classifier, open('NuSVC_classifier.p','wb'))
