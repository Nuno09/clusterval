'''
Module that imports datasets to be used in clustering.
Available datasets:
- khan - Microarray gene expression data set from Khan et al., 2001. Subset of 306 genes. Includes training and test set.
- vote.repub - Votes for Republican Candidate in Presidential Elections in each of the 50 states (1856 - 1976)
- animals - Attributes of 20 Animals

'''

import csv
import os
import numpy as np


def load_khan_train():

    train_khan = {'labels': [], 'data': []}

    with open(os.path.join(os.path.dirname(__file__), "train_khan.csv"), 'r') as file1:
        reader1 = csv.reader(file1)
        for line in reader1:
            train_khan['labels'].append(line[0])
            train_khan['data'].append(line[1:])

    train_khan['data'] = np.array(train_khan['data'])

    return train_khan



def load_khan_test():

    test_khan = {'labels': [], 'data': []}

    with open(os.path.join(os.path.dirname(__file__), "test_khan.csv"), 'r') as file2:
        reader2 = csv.reader(file2)
        for line in reader2:
            test_khan['labels'].append(line[0])
            test_khan['data'].append(line[1:])

    test_khan['data'] = np.array(test_khan['data'])

    return test_khan



def load_vote_repub():

    vote_repub = {'labels': [], 'data': []}

    with open(os.path.join(os.path.dirname(__file__), "vote_repub.csv"), 'r') as file3:
        reader3 = csv.reader(file3)
        next(reader3)
        for line in reader3:
            if 'NA' in line:
                for i, el in enumerate(line):
                    if el == 'NA':
                        line[i] = str(float(0))
            vote_repub['labels'].append(line[0])
            vote_repub['data'].append(line[1:])

    vote_repub['data'] = np.array(vote_repub['data'])

    return vote_repub


def load_animals():

    animals = {'labels': [], 'data': []}

    with open(os.path.join(os.path.dirname(__file__), "animals.csv"), 'r') as file4:
        reader4 = csv.reader(file4)
        next(reader4)
        for line in reader4:
            if 'NA' in line:
                for i, el in enumerate(line):
                    if el == 'NA':
                        line[i] = '0'
            animals['labels'].append(line[0])
            animals['data'].append(line[1:])

    animals['data'] = np.array(animals['data'])

    return animals
