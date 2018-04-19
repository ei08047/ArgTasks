#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ArgumentMiningCorpusReader
"""

import os
from abc import ABCMeta, abstractmethod



# Functions
class ArgumentMiningCorpusReader:
    __metaclass__= ABCMeta
    
    
    @abstractmethod
    def getArgumentDiagram(self, filepath, filename):
        """
        """
    
    
    @abstractmethod
    def graphToAIF(self, graph):
        """
        """
        
    
    
    # path= path to corpus
    # fileExtension= files extension (e.g. ".txt")
    def getAllFilenamesFromDir(self, path, fileExtension):
        
        filenames= []
        
        for textFile in os.listdir(path):
            if textFile.endswith(fileExtension):
                currentFileName= textFile.split(fileExtension)[0]
                filenames.append(currentFileName)
        return filenames
    
    



