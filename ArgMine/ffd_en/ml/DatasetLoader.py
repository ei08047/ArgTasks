#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DatasetLoader: build dataset for Machine Learning Task
"""

from sklearn.datasets.base import Bunch
from sklearn.utils import shuffle
import math
from abc import ABCMeta, abstractmethod

from utils.Parameters import Parameters

parameters= Parameters()
paths= parameters.paths
filenames= parameters.filenames


class DatasetLoader:
    __metaclass__= ABCMeta
    
    def __init__(self):
        print ("\n Loading Dataset ...\n")
        self.dataset= Bunch()
        (self.dataset).data= []
        (self.dataset).target= []
        (self.dataset).target_names= []
        # populate dataset
        self.addLearningInstancesToDataset()

    @abstractmethod
    def addLearningInstancesToDataset(self):
        """
        
        """
    
    @abstractmethod
    # output: bunch of the training set divided by percentage of articles "self.trainingSetPercentage" to include in the training set
    def getTainingTestSetSplit(self, trainingSetPercentageSplit= 0.8, randomStateSeed= 12345):
        """
        """

    def getCompleteDataset(self):
        b= Bunch()
        b.data= (self.dataset).data
        b.target= (self.dataset).target
        b.target_names= (self.dataset).target_names
        return b