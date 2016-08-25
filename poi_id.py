#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

#sys.path.append("E:/NanoDegreeDataAnalyst/git-repo/Udacity/tools")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
import numpy as np
import matplotlib.pyplot

#def name_fine(data,col,)

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features
features = ['salary','bonus']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#print data_dict.keys()

#Get total number of features and data points
person_data = pd.DataFrame(data_dict)   
person_data = person_data.transpose()


print "Total Number of Features", person_data.shape[1]
print "Total Number of DataPoints", person_data.shape[0]

#print person_data.head
#Clean the data replace string nan with numpy Nan






#person_data = pd.to_numeric(person_data,args=('coerce',))
person_data = person_data.apply(pd.to_numeric, errors='coerce')
person_data['email_address'] = person_data['email_address'].astype(str)

print "Data type of data frame", person_data.dtypes

#for val in list(person_data.columns.values):
    #if val != 'email_address':
        


print "Nan data in column\n",person_data.isnull().sum()


poi_count = person_data['poi'].loc[person_data['poi'] == True].count()
print "Number of POI",person_data['poi'].loc[person_data['poi'] == True].count()

print "Number of non POI", (person_data.shape[0] - poi_count)

print person_data.columns.values
#for data in data_dict:
    

max_bonus = 0
data = featureFormat(data_dict, features)  
for point in data:
    salary = point[0]
    bonus = point[1]
    #if bonus != 97343619.0:
        #matplotlib.pyplot.scatter( salary, bonus )
    if bonus > max_bonus:
        max_bonus = bonus
#data frame to search for max
#print person_data.iloc[0]
print "Maximum bonus",max_bonus
name_outlier = person_data[person_data['bonus'] == max_bonus].index.tolist()[0]

print "Person With Max bonus:",name_outlier

print "Details of the person with max bonus", data_dict[name_outlier]

print "Total rows:",len(data_dict)
new_dataset = removekey(data_dict,name_outlier)

print "Total rows after removal",len(new_dataset)

#my_dataset = 






#print max_bonus

#print "Person Information with highest bonus and salary\n",person_data.loc[person_data['bonus'] == max_bonus]

#Value with maximum bonus is actually total

"""
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show() 
"""
#person_data.to_csv('data.csv')
person_data.to_csv('data.csv')
print "\nCSV File Created"


name_outlier = []
"""
for key,val in data_dict.iteritems():
    if ((val['bonus'] =="NaN") or (val['salary'] =="NaN")):
        continue
    if((val['bonus'] > 5000000) and (val['salary'] > 1000000)):
        name_outlier.append(key)


print "Information of people in outlier",name_outlier

for name in name_outlier:
    print person_data.loc[name]
    print "\n\n"

"""







### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)