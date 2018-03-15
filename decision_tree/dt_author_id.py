#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score


###XXX: features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

print('features_train ==> ', len(features_train[0]))

clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")
t0 = time()
pred = clf.predict(features_test)
print("predict time:", round(time()-t0, 3), "s")

print("DecisionTree accuracy: %r" % accuracy_score(pred, labels_test))
print("10th: %r, 26th: %r, 50th: %r" % (pred[10], pred[26], pred[50]))
print("No. of predicted to be in the 'Chris'(1): %r" % sum(pred))
acc = accuracy_score(pred, labels_test) ### you fill this in!
### be sure to compute the accuracy on the test set

def submitAccuracies():
  return {"acc":round(acc,3)}

#########################################################


