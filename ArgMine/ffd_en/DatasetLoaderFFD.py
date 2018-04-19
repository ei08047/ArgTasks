#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DatasetLoaderFFD: build dataset for Fact Feel Detection
"""
import math

from pymongo import MongoClient

from sklearn.datasets.base import Bunch
from sklearn.utils import shuffle
import utils.Util as Util
from ffd_en.ml.DatasetLoader import DatasetLoader
from ffd_en.FactFeel import FactFeel
from utils.Parameters import Parameters
from nltk.tokenize import word_tokenize
parameters = Parameters()
paths = parameters.paths

class DatasetLoaderFFD(DatasetLoader):
    def addLearningInstancesToDataset(self):
        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        fact_feel_corpus = mongoClient.FACTFEELCorpus
        documentCollection = fact_feel_corpus.documents
        fact_feel = FactFeel()
        for fact_document in fact_feel.facts:
            raw_doc = fact_document.get_raw_text(fact_feel.path)
            tokens = word_tokenize(raw_doc)
            documentCollection.insert({'document_id': fact_document.get_doc_leaf() ,'raw': raw_doc, 'tokens':tokens})
            ((self.dataset).data).append(fact_document.get_doc_leaf())
            ((self.dataset).target).append(0)
        for feel_document in fact_feel.feels:
            raw_doc = feel_document.get_raw_text(fact_feel.path)
            tokens = word_tokenize(raw_doc.lower())
            documentCollection.insert({'document_id': feel_document.get_doc_leaf() ,'raw': raw_doc, 'tokens':tokens})
            ((self.dataset).data).append(feel_document.get_doc_leaf())
            ((self.dataset).target).append(1)
        # add target names to dataset bunch
        (self.dataset).target_names = ['fact', 'feel']
        # close connection
        mongoClient.close()

    # output: bunch of the training set divided by percentage of articles "self.trainingSetPercentage" to include in the training set
    def getTrainingTestSetSplit(self, trainingSetPercentageSplit=0.8, randomStateSeed=12345):
        trainingSet = Bunch()
        testSet = Bunch()

        articleIdsSet = shuffle(list(set([elem for elem in (self.dataset).data])), random_state=randomStateSeed)

        articlesIdsTestSet = articleIdsSet[
                             0:int(math.floor((((1.0 - trainingSetPercentageSplit) * len(articleIdsSet)) - 1)))]

        trainingSet.data = [elem for elem in (self.dataset).data if elem not in articlesIdsTestSet]

        trainingSet.target = [(self.dataset).target[elemIndex] for elemIndex in range(len((self.dataset).target)) if
                              (self.dataset).data[elemIndex] not in articlesIdsTestSet]

        trainingSet.target_names = (self.dataset).target_names

        testSet.data = [elem for elem in (self.dataset).data if elem in articlesIdsTestSet]

        testSet.target = [(self.dataset).target[elemIndex] for elemIndex in range(len((self.dataset).target)) if
                          (self.dataset).data[elemIndex] in articlesIdsTestSet]

        testSet.target_names = (self.dataset).target_names

        print('getTrainingTestSetSplit ## trainingSet: {}-{} | testSet: {}-{}'.format(len(trainingSet.data),len(trainingSet.target),len(testSet.data),len(testSet.target)))


        return (trainingSet, testSet)

    def determineSentenceTargetValue(self, sentence, annotation):
        currentSentenceIsArgument = False
        for node in annotation.nodes:
            if ((node.nodeInfo).nodeType == 'MajorClaim') or ((node.nodeInfo).nodeType == 'Claim') or (
                (node.nodeInfo).nodeType == 'Premise'):
                # TODO: there is a build in function that is faster and may do the same thing
                if Util.relaxedSubStringSimilarity((node.nodeInfo).text, sentence, 0.85):
                    currentSentenceIsArgument = True
                    break

        return currentSentenceIsArgument

    def getDatasetInfo(self):
        numberOfDocuments = 0
        numberOfFactStyleArguments = 0
        numberOfFeelStyleArguments = 0
        for i in range(0, len((self.dataset).data)):
            numberOfDocuments += 1
            if (self.dataset).target[i] == 1:
                numberOfFeelStyleArguments += 1
            elif (self.dataset).target[i] == 0:
                numberOfFactStyleArguments += 1

        print("no. fact arguments= " + str(numberOfFactStyleArguments))
        print("no. feel  arguments= " + str(numberOfFeelStyleArguments))
        print("total no. of documents= " + str(numberOfDocuments))

'''
### Test ###
myDatasetLoader = DatasetLoaderFFD()
myDatasetLoader.getDatasetInfo()
'''


