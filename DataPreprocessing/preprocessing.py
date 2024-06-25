import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from collections import Counter


class Preprocessing:
    attribute_names =["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","X24","X25","X26","X27","X28","X29","X30","X31","X32","X33","X34","X35","X36","X37","X38","X39","X40","X41","X42","X43","X44","X45","X46","X47","X48","X49","X50","X51","X52","X53","X54","X55","X56","X57","X58","X59","X60","X61","X62","X63","X64","class"]
    #def __init__(self):
    #    self.attribute_names =["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","X24","X25","X26","X27","X28","X29","X30","X31","X32","X33","X34","X35","X36","X37","X38","X39","X40","X41","X42","X43","X44","X45","X46","X47","X48","X49","X50","X51","X52","X53","X54","X55","X56","X57","X58","X59","X60","X61","X62","X63","X64","class"]
        
        
    def importData(year: str):
        X: pd.DataFrame
        y: pd.DataFrame

        arff_file = arff.loadarff('PolishBankruptcy/%syear.arff'%year)

        df = pd.DataFrame(arff_file[0])

        for i in range(len(df)):
            if df.loc[i, 'class'] == b'0':
                df.loc[i, 'class'] = 0.0
            else:
                df.loc[i, 'class'] = 1.0 

        data = df.dropna()
        
        return data
    
    def standardizeData(data: pd.DataFrame):
        scaler = preprocessing.StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data_scaled_transposed = data_scaled.transpose()
       
        attribute_dict = {}
        for i in range(len(data_scaled_transposed)):
            attribute_dict[Preprocessing.attribute_names[i]] = data_scaled_transposed[i]

        standardisedData = pd.DataFrame(attribute_dict)
        
        return standardisedData
    
    def  normalizeData(data: pd.DataFrame):
        normalizer = preprocessing.Normalizer()
        data_normalised = normalizer.fit_transform(data)
        data_normalised_transposed = data_normalised.transpose()
    
        normalised_dict = {}

        for i in range(len(data_normalised_transposed)):
            normalised_dict[Preprocessing.attribute_names[i]] = data_normalised_transposed[i]


        normalisedData = pd.DataFrame(normalised_dict)
        
        return normalisedData
    
    def normalizeAndStandardizeData(data: pd.DataFrame):
        standardized_data = Preprocessing.standardizeData(data)
        normalised_standardised_data = Preprocessing.normalizeData(standardized_data)

        return normalised_standardised_data
    
    def discretizeData(rawData: pd.DataFrame, processedData: pd.DataFrame):
        
        atttributes = processedData.columns.values.tolist()

        est = preprocessing.KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile', subsample=200_000)
        est.fit(processedData)
        PreprocessedData_tranformed = est.transform(processedData).transpose()
        attribute_dict = {}
        for i in range(len(PreprocessedData_tranformed)):
            attribute_dict[atttributes[i]] = PreprocessedData_tranformed[i]

        attribute_dict[Preprocessing.attribute_names[len(Preprocessing.attribute_names)-1]] = pd.cut(rawData['class'],2,labels=[0,1])

        discretizedData = pd.DataFrame(attribute_dict)
        
        keeping_attributes = [
            "X23",
            "X1",
            "X7",
            "X49",
            "X9",
            "X3",
            "X2",
            "X59",
            "X6",
            "X4",
            "X51",
            "X8",
            "class"
        ]
        
        most_common_attributes = ['X52', 'X40', 'X25', 'X5', 'X9', 'X13', 'X58', 'X15', 'X27', 'X36', 'X22', 'X42', 'X31', 'X48', 'X32', 'X57', 'X12', 'X35', 'X6', 'X53', 'X16', 'X11', 'X55', 'X14', 'X29', 'X41','class']
        
        most_common_attributes_all_years = ['X16', 'X52', 'X32', 'X28', 'X5', 'X40', 'X9', 'X11', 'X59', 'X23', 'X25', 'X55', 'X17', 'X14', 'X29', 'X13', 'X58', 'X30', 'X57', 'X56', 'X40', 'X15', 'X27', 'X5', 'X25', 'X36', 'X22', 'X42', 'X31', 'X13', 'X12', 'X35', 'X9', 'X58', 'X11', 'X48', 'X52', 'X57', 'X55', 'X6', 'X15', 'X22', 'X52', 'X27', 'X40', 'X5', 'X25', 'X31', 'X12', 'X42', 'X13', 'X53', 'X57', 'X37', 'X48', 'X6', 'X35', 'X41', 'X32', 'X36', 'X22', 'X52', 'X15', 'X25', 'X27', 'X40', 'X58', 'X42', 'X13', 'X36', 'X31', 'X5', 'X53', 'X6', 'X35', 'X48', 'X9', 'X24', 'X38', 'X29', 'X25', 'X22', 'X27', 'X15', 'X52', 'X53', 'X14', 'X40', 'X42', 'X36', 'X54', 'X12', 'X58', 'X41', 'X44', 'X48', 'X9', 'X31', 'X32', 'X16', 'class']
        
        #for attribute in discretizedData.columns.values.tolist():
        #    if attribute not in most_common_attributes_all_years:
        #        del discretizedData[attribute]


        return discretizedData
    
    def splitData(data: pd.DataFrame):
        processed_bankrupt = data.loc[data["class"] == 1]
        processed_nonbunkrupt = data.loc[data["class"] == 0]
        processed_bankrupt_train = processed_bankrupt[0:(floor(0.75*processed_bankrupt.shape[0]))]
        processed_nonbankrupt_train = processed_nonbunkrupt[0:(floor(0.75*processed_nonbunkrupt.shape[0]))]

        processed_bankrupt_test = processed_bankrupt[(floor(0.75*processed_bankrupt.shape[0]))+1:processed_bankrupt.shape[0]]
        processed_nonbankrupt_test = processed_nonbunkrupt[(floor(0.75*processed_nonbunkrupt.shape[0]))+1:processed_nonbunkrupt.shape[0]]
        training_data = pd.concat([processed_nonbankrupt_train,processed_bankrupt_train])

        testing_data = pd.concat([processed_nonbankrupt_test,processed_bankrupt_test])
        testing_targets = testing_data["class"]
        del testing_data["class"]

        testing_evidence_list = []
        for i in range(len(testing_data)):
            testing_evidence_dict = {}
            for z in range(len(testing_data.columns.values.tolist())):
                testing_evidence_dict[testing_data.columns.values.tolist()[z]] = testing_data[testing_data.columns.values.tolist()[z]].iloc[i]
            testing_evidence_list.append(testing_evidence_dict)

        return training_data, testing_data, testing_targets, testing_evidence_list
    
    