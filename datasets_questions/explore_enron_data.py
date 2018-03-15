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
# =============================================================================
#   1) Executar o arquivo ..\final_project/dos2unix.txt
#   2) Alterar o nome do arquivo para o novo arquivo criado: final_project_dataset_unix.pkl
# =============================================================================

import pickle
#import cPickle as pickle
enron_data = pickle.load(open('../final_project/final_project_dataset_unix.pkl', 'rb'))

for label in enron_data:
    # print(enron_data["SKILLING JEFFREY K"])
    print(enron_data[label])
