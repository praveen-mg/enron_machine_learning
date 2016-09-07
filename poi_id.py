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
from sklearn.metrics import accuracy_score

    
def convert_dict(data_frame):
    """
    convert data frame into dictionary
    """
    temp_df = data_frame
    print "Shape of DF called",data_frame.shape
    temp_df = temp_df.fillna('NaN')
    temp_df = temp_df.transpose()
    dataset = temp_df.to_dict()
    return dataset
    
if __name__ == "__main__":
    
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    # You will need to use more features
    features_list = ['poi','salary','bonus'] 
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
    print "Nan data in column\n",person_data.isnull().sum()
    poi_count = person_data['poi'].loc[person_data['poi'] == True].count()
    print "Number of POI",person_data['poi'].loc[person_data['poi'] == True] \
        .count()
    print "Number of non POI", (person_data.shape[0] - poi_count)
    print person_data.columns.values

        
    ### Task 2: Remove outliers
    #Analysis Based on Plotting
        
        
    matplotlib.pyplot.scatter( person_data['salary'], person_data['bonus'])
    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel("bonus")
    matplotlib.pyplot.title("Before Removing the outlier")
    matplotlib.pyplot.show() 
    
    #Find Outlier and remove it
          
    max_bonus = person_data['bonus'].max()
    data = featureFormat(data_dict, features)
    print "Maximum bonus",max_bonus
    name_outlier = person_data[person_data['bonus'] == \
        max_bonus].index.tolist()[0]
    print "Person With Max bonus:",name_outlier
    print "Details of the person with max bonus", data_dict[name_outlier]
    print "Total rows:",len(data_dict)
    person_data = person_data[person_data['bonus'] != max_bonus]
    my_dataset = convert_dict(person_data)
    matplotlib.pyplot.scatter( person_data['salary'], person_data['bonus'],\
        c=person_data['poi'])
    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel("bonus")
    matplotlib.pyplot.title("After Removing the outlier")
    matplotlib.pyplot.show() 
    
    #Check For other Outliers
    
    other_outlier = person_data[(person_data['bonus'] > 5000000) & \
        (person_data['salary'] > 1000000)]
    payment_max = person_data['total_payments'].max()
    name_payment_outlier = person_data[person_data['total_payments'] == 
        payment_max].index.tolist()[0]    
#    for  outlier in other_outlier.index.tolist():
#        person_data = person_data[person_data.index != outlier]
    print "Person data shape after other outlier removal",person_data.shape
    print "Payment Outlier",name_payment_outlier
    print "Payment Outlier Details",data_dict[name_payment_outlier]
    print "Outlier Defferal Income"
    deferral_payment_outliers = person_data['deferral_payments'].min()
    name_deferral_payment_outliers = person_data[ \
        person_data['deferral_payments']<0].index.tolist()
    print name_deferral_payment_outliers
    print"Person data before removing deferral_payments outlier", \
        person_data.shape   
    person_data = person_data[person_data.deferral_payments != \
        deferral_payment_outliers]
    print"Person data after removing deferral_payments outlier", \
        person_data.shape 

    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
   
    person_data['to_ratio'] = \
        person_data['from_poi_to_this_person'] /person_data['to_messages']
    person_data['from_ratio'] = \
        person_data['from_this_person_to_poi']/person_data['from_messages']
    person_data['deferred_income'] = -person_data['deferred_income']
    
    
    #Create CSV file after removing outliers
    
    person_data.to_csv('data.csv')
    print "\nCSV File Created"
    my_dataset = convert_dict(person_data)
    print "my dataset:",len(my_dataset)
    
     
    #Plot Ratio of from and to messages
    
    matplotlib.pyplot.scatter( person_data['to_ratio'], 
                              person_data['from_ratio'],c=person_data['poi'])
    matplotlib.pyplot.xlabel("To Ratio")
    matplotlib.pyplot.ylabel("From Ratio")
    matplotlib.pyplot.title("Relative Values Plot")
    matplotlib.pyplot.show() 
    
    #create feature list with different combination of features
    
    
    features_list = ['poi']
    #features_list.append('to_ratio')
    #features_list.append('salary')
    #features_list.append('bonus')
    features_list.append('from_ratio')
    #features_list.append('exercised_stock_options')
    #features_list.append('total_stock_value')
    features_list.append('other') 
    features_list.append('expenses')
    features_list.append('shared_receipt_with_poi')   
    #features_list.append('restricted_stock')
    #features_list.append('total_payments')  
    print features_list
   
    
    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    
    
    ### Task 5: Tune your classifier to achieve better than 
        ###.3 precision and recall 
    ### using our testing script. Check the tester.py script in 
        ###the final project
    ### folder for details on the evaluation method, especially 
        ###the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/
        ###generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    
    from sklearn.naive_bayes import GaussianNB
    from sklearn import tree
    #clf = GaussianNB()
    #clf = tree.DecisionTreeClassifier()        
    clf = tree.DecisionTreeClassifier(min_samples_split=16,min_samples_leaf=2,
                                      criterion='entropy')
                                      
                            
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.4, random_state=42)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    score = accuracy_score(pred,labels_test)
    print score
#    features_list_name = features_list[1:]
#    clf.fit(features_train,labels_train)
#    print zip(features_list_name,clf.feature_importances_)
    
    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, 
        ###but make sure
    ### that the version of poi_id.py that you submit can be run 
        ###on its own and
    ### generates the necessary .pkl files for validating your results.
    
    dump_classifier_and_data(clf, my_dataset, features_list)