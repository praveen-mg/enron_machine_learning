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
    
def float_to_int(val):
    if type(val) is str:
        return val
    else:
        if pd.isnull(val):
            return val
        elif type(val) is float:
            return int(val)
        else:
            return val
    
def convert_dict(data_frame):
    
    #temp_df = data_frame.astype(int,errors='coerce')
    #print data_frame.head()
    temp_df = data_frame
    #for col in temp_df.columns.values:
        #print col
        #temp_df[col] = temp_df[col].map(float_to_int)
#        for val in data_frame[col]:
#            if not pd.isnull(val):
    
    #temp_df = data_frame.apply(float_to_int)
    #temp_df = temp_df.applymap(float_to_int)
    #print "\n\n\nTEMP DF"
    #print temp_df.head()
    print "Shape of DF called",data_frame.shape
    temp_df = temp_df.fillna('NaN')
    temp_df = temp_df.transpose()
    dataset = temp_df.to_dict()
#    for key in dataset:
#        for key1,val in dataset[key].iteritems():
#            dataset[key][key1] =  float_to_int(val)
   
    return dataset
    
if __name__ == "__main__":
    
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".

    features_list = ['poi','salary'] # You will need to use more features
    features = ['salary','bonus']
    
    ### Load the dictionary containing the dataset
    
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
        
    #Convert data into dataframe for easy processing
    #Get total number of features and data points
    
    person_data = pd.DataFrame(data_dict)   
    person_data = person_data.transpose()
    print "Total Number of Features", person_data.shape[1]
    print "Total Number of DataPoints", person_data.shape[0]
    
     #Writing data to CSV
    
    person_data.to_csv('data.csv')
    
    print "\nCSV File Created"
    #Clean the data replace string nan with numpy Nan
    
    person_data = person_data.apply(pd.to_numeric, errors='coerce')
    person_data['email_address'] = person_data['email_address'].astype(str)
    #print "Data type of data frame", person_data.dtypes
    #print "Nan data in column\n",person_data.isnull().sum()
    poi_count = person_data['poi'].loc[person_data['poi'] == True].count()
    #print "Number of POI",person_data['poi'].loc[person_data['poi'] == True].count()
    #print "Number of non POI", (person_data.shape[0] - poi_count)
    print person_data.columns.values

        
    ### Task 2: Remove outliers
  
    """
    for point in data:
        salary = point[0]
        bonus = point[1]
        #if bonus != 97343619.0:
        if bonus != max_bonus:
            matplotlib.pyplot.scatter( salary, bonus )
            
    """
    #Analysis Based on Plotting
    
    
    
    matplotlib.pyplot.scatter( person_data['salary'], person_data['bonus'])
    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel("bonus")
    matplotlib.pyplot.title("Before Removing the outlier")
    matplotlib.pyplot.show() 
    
    #Find Outlier and remove it
          
    max_bonus = person_data['bonus'].max()
    data = featureFormat(data_dict, features)
    #print "Maximum bonus",max_bonus
    name_outlier = person_data[person_data['bonus'] == max_bonus].index.tolist()[0]
    #print "Person With Max bonus:",name_outlier
    #print "Details of the person with max bonus", data_dict[name_outlier]
    #print "Total rows:",len(data_dict)
    #my_dataset = removekey(data_dict,name_outlier)
    data_dict  = removekey(data_dict,name_outlier)
    my_dataset = data_dict
    #print "Rows Before removal",person_data.shape
    person_data = person_data[person_data['bonus'] != max_bonus]
    #print "Rows Before Call",person_data.shape
    my_dataset = convert_dict(person_data)
    #print "Return Value:%d"%  cmp (my_dataset, my_dataset1)
    matplotlib.pyplot.scatter( person_data['salary'], person_data['bonus'],c=person_data['poi'])
    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel("bonus")
    matplotlib.pyplot.title("After Removing the outlier")
    matplotlib.pyplot.show() 
    
    #Check For other Outliers
    
    other_outlier = person_data[(person_data['bonus'] > 5000000) & (person_data['salary'] > 1000000)]
    #print other_outlier
#    print "my dataset:",len(my_dataset)
#    print "my dataset1:",len(my_dataset1)
#    for key in my_dataset:
#        if key in my_dataset1:
#            continue
#        else:
#            print "Key is miss",key
#        for key1 in my_dataset[key]:
#            if my_dataset[key][key1] == my_dataset1[key][key1]:
#                continue
#            else:
#                print "key",key
#                print "key1",key1
#                print "Val",my_dataset[key][key1]
#                print "Val1",my_dataset1[key][key1]
#            if val == val1:
#                continue
#            else:
#                print "Val",val
#                print "Val1",val1
#                print "key",key
#                print "key1",key1
#    
    
    
    
    
    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
   
    person_data['to_ratio'] = person_data['from_poi_to_this_person'] /person_data['to_messages']
    person_data['from_ratio'] = person_data['from_this_person_to_poi']/person_data['from_messages']
    my_dataset = convert_dict(person_data)
    print "my dataset:",len(my_dataset)
    #features_list = []
    features_list.append('to_ratio')
    features_list.append('from_ratio')
    features_list.append('deferral_payments')
    features_list.append('deferred_income')
    features_list.append('expenses')
    #features_list.append('restricted_stock_deferred')    
    
 
     
    
    print features_list
    
    #Plot Ratio of from and to messages
    
    matplotlib.pyplot.scatter( person_data['to_ratio'], person_data['from_ratio'],c=person_data['poi'])
    matplotlib.pyplot.xlabel("To Ratio")
    matplotlib.pyplot.ylabel("From Ratio")
    matplotlib.pyplot.title("Relative Values Plot")
    matplotlib.pyplot.show() 
    
    
    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    
    # Provided to give you a starting point. Try a variety of classifiers.
    #from sklearn.naive_bayes import GaussianNB
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split = 4)
    
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