#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'long_term_incentive', 'deferred_income', 
                 'deferral_payments', 'other', 'expenses', 'exercised_stock_options', 'total_stock_value',
                 'from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset.
### Note: For Python 3, pickle.load gives an error. Followed suggestion from this link:
### https://github.com/udacity/ud120-projects/issues/46
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

feature_df = pd.DataFrame.from_dict(data_dict, orient='index')

# Data exploration
print ("Data set size: ", feature_df.shape)
print (feature_df.describe(include='all'))

print ('POI vs. Non-POI: ', feature_df['poi'].value_counts())


feature_df.replace('NaN', np.nan, inplace = True)
print (feature_df.isnull().sum())
    
### Task 2: Remove outliers
feature_df.plot.scatter(x = 'salary', y = 'bonus')
plt.show()

outlier = feature_df['salary'].idxmax()

feature_df = feature_df.drop(outlier)

feature_df.plot.scatter(x = 'salary', y = 'bonus')
feature_df.plot.scatter(x = 'salary', y = 'long_term_incentive')
feature_df.plot.scatter(x = 'salary', y = 'deferred_income')
feature_df.plot.scatter(x = 'salary', y = 'deferral_payments')
feature_df.plot.scatter(x = 'salary', y = 'other')
feature_df.plot.scatter(x = 'salary', y = 'expenses')
feature_df.plot.scatter(x = 'salary', y = 'exercised_stock_options')
feature_df.plot.scatter(x = 'salary', y = 'total_stock_value')
plt.show()
    
### Task 3: Create new feature(s)
# Feature 1: fraction_messages_to_poi
feature_df['fraction_messages_to_poi'] = feature_df.from_this_person_to_poi/feature_df.from_messages

# Feature 2: fraction_messages_from_poi
feature_df['fraction_messages_from_poi'] = feature_df.from_poi_to_this_person/feature_df.to_messages

# Feature 3: total stock value vs. salary
feature_df['stockvssalary'] = feature_df.total_stock_value/feature_df.salary

# Feature 4: other payments vs. salary
feature_df['othervssalary'] = feature_df.other/feature_df.salary

feature_df = feature_df.fillna(value=0)
features_list.extend(['fraction_messages_from_poi','fraction_messages_to_poi', 'stockvssalary', 'othervssalary'])

### Store to my_dataset for easy export below.
my_dataset = feature_df.to_dict('index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_list_minus_label = features_list[1:len(features_list)]

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=20)

# Trying K best to select the most relevant features
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
selector.fit(features_train, labels_train)

feature_scores = ['%.2f' % elem for elem in selector.scores_]

# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  selector.pvalues_]

# Get SelectKBest feature names, whose indices are stored in 'selector.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in selector.get_support(indices=True)]

# Sort the tuple by score, in reverse order

features_selected_tuple = sorted(features_selected_tuple, key=lambda 
feature: float(feature[1]) , reverse=True)

print (' ')
print ('KBest: Selected Features, Scores, P-Values')
print (features_selected_tuple)

# Transform train and test data according to KBest
features_train = selector.transform(features_train)
features_test = selector.transform(features_test)


features_list = [row[0] for row in features_selected_tuple]
features_list.insert(0, "poi") #poi has to be first in the feature list
print (features_list)

# end of K best

### Task 4: Try a variety of classifiers

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tester import test_classifier

# 1. SVM
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

parameters = {'C':[1, 10, 100], 'gamma':[0.001, 0.01, 0.1]}
clf = make_pipeline(StandardScaler(), GridSearchCV(SVC(kernel='rbf', class_weight='balanced', random_state=20), parameters))
#clf = make_pipeline(StandardScaler(), SVC(C=100.0, kernel='rbf', class_weight='balanced', gamma=0.01, random_state=42))
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)
print ("Confusion matrix for SVM : ")
print (confusion_matrix(labels_test, labels_pred, labels=[0.0, 1.0]))
print ("Tester results for SVM: ")
test_classifier(clf, my_dataset, features_list)

# 2. Decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2, max_features=10, max_depth=9, random_state=20)
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Decision tree feature ranking:")
for f in range(features_train.shape[1]):
    print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], features_list_minus_label[indices[f]]))

print ("Confusion matrix for Decision tree classifier: ")
print (confusion_matrix(labels_test, labels_pred, labels=[0.0, 1.0]))
print ("Tester results for Decision tree classifier: ")
test_classifier(clf, my_dataset, features_list)

# 3. Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)
print ("Confusion matrix for Gaussian NB: ")
print (confusion_matrix(labels_test, labels_pred, labels=[0.0, 1.0]))
print ("Tester results for Gaussian NB: ")
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)