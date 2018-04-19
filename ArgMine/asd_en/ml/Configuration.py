#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration:
"""

from abc import ABCMeta, abstractmethod


class Configuration:
    __metaclass__= ABCMeta
    
    def __init__(self, fixed= True, type= "best"):
        
        print ("\n Loading Configurations ...\n")
        
        self.fixed= fixed
        self.type= type
        
    
    @abstractmethod
    def loadFeatureSet(self):
        """
        """
    
    
    @abstractmethod
    def loadFeaturesConfigs(self):
        """
        """
    
    
    @abstractmethod
    def loadClassifiersConfigs(self):
        """
        """
    
    
    @abstractmethod
    def loadFilterMethodsConfigs(self):
        """
        """
    
    
    def getCompleteParametersSet(self, featuresConfig, filtersConfig, classifiersConfig):
        completeParametersSet= {}
        
        completeParametersSet.update(featuresConfig)
        completeParametersSet.update(filtersConfig)
        completeParametersSet.update(classifiersConfig)
        
        return completeParametersSet
        
    
    
    
    