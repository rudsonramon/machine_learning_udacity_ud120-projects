#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#enron_data = pickle.load(open('../final_project/final_project_dataset_unix.pkl', 'rb'))

def open_dataset(url, param_permission='rb'):
    """
       open the file that will be used in this project.
    """
    enron_data = pickle.load(open(url, param_permission))
    return enron_data

def extract_data(data_file):
    """
        TRANSFORM A PICKLE FILE IN THE PANDAS DATA FRAME
    """
    df = pd.DataFrame.from_dict(data_file)
    # print(df)
    # df.info()
    # df.head()
    return df

## OPEN THE FILE
######  USING A PIVOT WAY TO TRY GETTING BETTER VIEW
enron_data = open_dataset('../final_project/final_project_dataset_unix.pkl', 'rb')

df = extract_data(enron_data)


# df = enron_data.iloc[:5 , :3]

# stacked = df.stack()

# print('stacked: ' + '\r\n', stacked)
## unstacked = stacked.unstack()
## print('unstacked: ' + '\r\n', unstacked)

enron_df = pd.DataFrame.from_dict(df)

###XXX: USING PANDAS DATA FRAME WITH ILOC TO ORGANIZE THE DATASET
# enron_df.fillna(0, inplace=True) ##   fill the NAN values with zero '0'
# enron_df['BADUM JAMES P'] = enron_df['BADUM JAMES P'].fillna(0, inplace=True)
# enron_df.fillna(0)

employee = {'ALLEN PHILLIP K': 0} ## EMPLOYEE_NAME replaces the NAN values for zero.
enron_print = enron_df.fillna(value=employee)
enron_print = enron_print.filter(items=['ALLEN PHILLIP K', 'BADUM JAMES P']) ## filter columns 
enron_print = enron_print.filter(like='salary', axis=0) ## filter rows

# print(enron_print)
# enron_print['BADUM JAMES P'].replace('nan', '')
# enron_print['Max'] = df[['ALLEN PHILLIP K','BADUM JAMES P']].idxmax(axis=1)

enron_df = enron_df.iloc[16:17,:]
# print(enron_print)
# print('\n\r', 'print(enron_df.describe())==>>', enron_df.describe())
# print('\n')
# print('\n\r', 'print(enron_print.describe())==>>', enron_print.describe())
print(enron_df)
enron_df.info()

enron_stacked = enron_df.stack()
# print('enron_stacked==>>', enron_stacked)
print(enron_stacked)
# enron_stacked['salary'].astype(int)
old_label = 0
old_label = int(old_label)

for key, label in enron_stacked.items():
    if label != np.nan:
        if int(label) > int(old_label):
            big_salary = label
            print(big_salary)
        pass
    else:
        pass

    old_label = int(label)
    print('key==>>', key, 'label==>>', label)

    pass


"""
enron_df = df.iloc[[0], [0, 1]]

for key, label in enron_df.items():
    print('key ==>> ', key, 'label ==>> ', label)


red_patch = mpatches.Patch(color='red', label=key)
plt.legend(handles=[red_patch])
##  subplot(211) ==>> DEFINE THE UP POSITION OF THE PLOT
# plt.subplot(211)
##  label="Line 1", linestyle='--' DEFINE THE TYPE OF THE LINE
# .plot(kind='bar')
line1 = plt.plot(enron_df, linestyle='--', label=key) ## , marker='o', linestyle='--'

plt.show()

print('iloc: ' + '\r\n', enron_df)
"""

    ############# ALTERNATIVE SOLUTION TO SEE THE RESULTS
    # for feature, label in data_file.items():
    #     # print(enron_data["SKILLING JEFFREY K"])
    #     # print(enron_data[label])
    #     # return data_file[label]["email_address"]
    #     # print('feature ==>> ', feature, 'label ==>> ', label)
        
    #     print('feature ==>> ', feature, 'df ==>>', df)
    #     for item_label, item_description in label.items():
    #         return print('item_label ==>> ', item_label, 'item_description ==>> ', item_description)
    ######################################################

# Start processing the data

# RECEIVE THE CONTENT OF THE FILE FROM THE METHOD ==>> open_dataset(url, param_permission='rb')

##################################################################
##################################################################
##################################################################
# enron_data = open_dataset('../final_project/final_project_dataset_unix.pkl', 'rb')
## from final_project.poi_email_addresses import *
#def open_text_file(url, param_permission):
#    fp = open(url, param_permission)
#    return fp
## How many data points (people) are in the dataset?
#len(enron_data)
#len(enron_data.keys())
## print('len(enron_data)==>>', len(enron_data))
## print('len(enron_data.keys())==>>', len(enron_data.keys()))


## How Many POIs Exist?
#url = 'C:/Users/rudson.r.rodrigues/PythonProjects/Udacity/Intro Machine Learn/ud120-projects/final_project/poi_names.txt'
#poi_text = url
#poi_names = open(poi_text, 'r')
#fr = poi_names.readlines()
#len(fr[2:])
#poi_names.close()

## print poi_names.read()

#num_lines = sum(1 for line in open(poi_text))

# def file_len(fname):
    # with open(fname) as f:
        # for i, l in enumerate(f):
            # pass
    # return i + 1

# file_len(url)
# print('file_len(url)==>>',  file_len(url))


## poi_names = open_dataset('../final_project/poi_names.txt', 'rb')
## poi_mail_list = poiEmails()
## print('poi_names==>', poi_names)
## print('poi_e-mail_list==>', poi_mail_list)
## EXTRACT ALL THE DATA FROM THE METHOD ==>> extract_data(enron_data)
## print('enron_data ==>>', enron_data["SKILLING JEFFREY K"]["bonus"])

## feature_train, label_train = extract_data(enron_data)
#label_train_df = extract_data(enron_data)
## print('FILE CONTENT ==>> ', extract_data(enron_data))
## print('feature_train ==>> ', feature_train, 'label_train==>> ', label_train)

## label_train.info()
## label_train.pivot_table(index='ALLEN PHILLIP K', columns='ALLEN PHILLIP K', values='deferred_income', aggfunc='mean')
## label_train = label_train.iloc[0:3,:2]
## label_train = label_train[['ALLEN PHILLIP K', 'BAZELIDES PHILIP J']]
## print(label_train)
## import matplotlib.pyplot as plt
## plt.plot(label_train)
## plt.show()
## label_train = label_train.iloc[13:14,:]
#label_train = label_train_df.loc['poi',:]
#count_poi = 0
#for poi in label_train_df:
#    if poi == True:
#        count_poi += 1
#        pass
#    # print('poi==>> ', poi, 'count_poi==>> ', count_poi)
#print('GRAND TOTAL ==> count_poi==>> ', count_poi)

#count_stock = label_train_df[['PRENTICE JAMES'][0]]
#for stock_label, total_stock_value in count_stock.items():
#    if stock_label == 'total_stock_value':
#        print('PRENTICE JAMES:','total_stock_value==>>', total_stock_value)
#        pass
#    pass

#wesley_colwell = label_train_df[['COLWELL WESLEY'][0]]
#for stock_label, total_stock_value in wesley_colwell.items():
#    if stock_label == 'from_this_person_to_poi':
#        print('WESLEY COLWELL:','from_this_person_to_poi==>>', total_stock_value)
#        pass
#    pass

#jeffrey_k_skilling = label_train_df[['SKILLING JEFFREY K'][0]]
#for stock_label, total_stock_value in jeffrey_k_skilling.items():
#    if stock_label == 'exercised_stock_options':
#        print('JEFFREY K SKILLING:','exercised_stock_options==>>', total_stock_value)
#        pass
#    pass
# How much money did that person get?
#print("['SKILLING JEFFREY K']['total_payments']", label_train_df['SKILLING JEFFREY K']['total_payments'])
#print("['FASTOW ANDREW S']['total_payments']", label_train_df['FASTOW ANDREW S']['total_payments'])
#print("['LAY KENNETH L']['total_payments']", label_train_df['LAY KENNETH L']['total_payments'])

## How is an unfilled feature denoted?
#label_train_df['FASTOW ANDREW S']['deferral_payments']

## How many folks in this dataset have a quantified salary?
## What about a known email address?
#count_salary = 0
#count_email = 0
#for key in label_train_df.keys():
#    if label_train_df[key]['salary'] != 'NaN':
#        count_salary+=1
#    if label_train_df[key]['email_address'] != 'NaN':
#        count_email+=1
#print('count_salary==>>', count_salary)
#print('count_email==>', count_email)

## How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments?
## What percentage of people in the dataset as a whole is this?
#count_NaN_tp = 0
#for key in label_train_df.keys():
#    if label_train_df[key]['total_payments'] == 'NaN':
#        count_NaN_tp+=1
#print('count_NaN_tp==>>', count_NaN_tp)
#print('float(count_NaN_tp)/len(label_train_df.keys())==>>', float(count_NaN_tp)/len(label_train_df.keys()))

## How many POIs in the E+F dataset have “NaN” for their total payments? 
## What percentage of POI’s as a whole is this?   
#count_NaN_tp = 0
#for key in label_train_df.keys():
#    if label_train_df[key]['total_payments'] == 'NaN' and label_train_df[key]['poi'] == True :
#        count_NaN_tp+=1
#        print('count_NaN_tp+=1==>>', count_NaN_tp)

#print('count_NaN_tp==>>', count_NaN_tp)
#print('float(count_NaN_tp)/len(label_train_df.keys())==>>', float(count_NaN_tp)/len(label_train_df.keys()))

#print('label_train_df.iloc[:, :]==>', label_train_df.iloc[:, :])
## print('count_stock==>>', count_stock)
## for count_poi in label_train.iloc[0:10,]:
#    # count_poi +=1
## print('count_poi ==>', count_poi)
## count_poi = label_train.iloc[-8,:] == True
## print('label_train==>> ', label_train)
## print('count_poi ==>', count_poi)
## print(label_train)
## poi = []
## def find_poi(label_train)
