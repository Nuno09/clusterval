'''
Module that imports datasets to be used in clustering.
Available datasets:
- khan - Microarray gene expression data set from Khan et al., 2001. Subset of 306 genes. Includes training and test set.
- vote.repub - Votes for Republican Candidate in Presidential Elections in each of the 51 states (1856 - 1976)
- animals - Attributes of 20 Animals

'''

import csv
import os

class ImportDatasets:
    def __init__(self):

        train_khan = []
        test_khan = []
        vote_repub = []
        animals = []

        with open(os.path.join(os.path.dirname(__file__), "train_khan.csv"), 'r') as file1:
            reader1 = csv.reader(file1)
            for line in reader1:
                train_khan.append(line[1:])

        with open(os.path.join(os.path.dirname(__file__), "test_khan.csv"), 'r') as file2:
            reader2 = csv.reader(file2)
            for line in reader2:
                test_khan.append(line[1:])

        with open(os.path.join(os.path.dirname(__file__), "vote_repub.csv"), 'r') as file3:
            reader3 = csv.reader(file3)
            next(reader3)
            for line in reader3:
                if 'NA' in line:
                    for i, el in enumerate(line):
                        if el == 'NA':
                            line[i] = str(float(0))
                vote_repub.append(line[1:])

        with open(os.path.join(os.path.dirname(__file__), "animals.csv"), 'r') as file4:
            reader4 = csv.reader(file4)
            next(reader4)
            for line in reader4:
                if 'NA' in line:
                    for i, el in enumerate(line):
                        if el == 'NA':
                            line[i] = '0'
                animals.append(line[1:])


        self.khan_train = train_khan
        self.khan_test = test_khan
        self.vote_repub = vote_repub
        self.animals = animals



