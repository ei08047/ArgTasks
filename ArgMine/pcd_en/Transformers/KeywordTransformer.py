#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KeywordTransformer: Custom Transformer with the purpose to create the keywordsmodel features. 
A Transformer is a sklearn object that should implement the following methods: fit, 
transform and fit_transform (inheriting from TransformerMixin implements this method for free).
Generally, they accept a matrix as input and return a matrix of the same shape as output ( 
with the corresponding features values for each input element).
In this particular case, it receives as input a matrix with dimensions 1:X where each element 
X is a proposition. It outputs a matrix with dimension X:Y where X corresponds to the total 
number of learning instances received as input and the dimension in Y depends on the 'featureSetConfiguration' parameter
"""

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

from pymongo import MongoClient

from utils.Parameters import Parameters
parameters= Parameters()


class KeywordTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, featureSetConfiguration= 2):
        self.featureSetConfiguration= featureSetConfiguration
        
        premisseKeywordsFile= open(parameters.paths["keywords_en"] + '/' + parameters.filenames["premiseKeywords"],'r')
        
        self.premisseKeywordsList= []
        
        for word in premisseKeywordsFile:
            self.premisseKeywordsList.append(word.rstrip("\n")) # .decode("utf-8")
            
        conclusionKeywordsFile= open(parameters.paths["keywords_en"] + '/' + parameters.filenames["conclusionKeywords"],'r')
        
        self.conclusionKeywordsList= []
        
        for word in conclusionKeywordsFile:
            self.conclusionKeywordsList.append(word.rstrip("\n")) # .decode("utf-8")
        
        
        self.argKeywordsList= self.premisseKeywordsList + self.conclusionKeywordsList

    def transform(self, X, **transform_params):
        # X -> corresponds to the data -> array of tuples (articleId, sentenceId), where each element corresponds to a specific sentence from an article
        
        self.featureArray= []
        
        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        
        aaecCorpus = mongoClient.AAECCorpus
        
        # Sentence's table
        sentenceCollection= aaecCorpus.sentence
        for sentenceTuple in X:
            # current learning instance info from database
            currentSentence= sentenceCollection.find_one({"$and":[{"articleId": int(sentenceTuple[0])}, {"sentenceId": int(sentenceTuple[1])}]})
            premiseCounter= 0
            conclusionCounter= 0
            for currentToken in currentSentence["tokens"]:
                # argumentative keywords counters
                if currentToken["lemma"] in self.premisseKeywordsList:
                    premiseCounter= premiseCounter + 1
                if currentToken["lemma"] in self.conclusionKeywordsList:
                    conclusionCounter= conclusionCounter + 1
            #TODO: some code with PBL restriction was removed in the current version. See if we should get it back!
            
            if self.featureSetConfiguration == 0:
                # Transformer is not active
                self.featureArray.append([0])
            elif self.featureSetConfiguration == 1:
                # Boolean Feature indicating if current sentence has (1) or hasn't (0) keyword
                if premiseCounter + conclusionCounter > 0:
                    self.featureArray.append([1])
                else:
                    self.featureArray.append([0])
                
            elif self.featureSetConfiguration == 2:
                # [number of premise keywordsmodel in current sentence, number of conclusion keywordsmodel in current sentence]
                #self.featureArray.append([premiseCounter, conclusionCounter])
                if premiseCounter > 0:
                    premiseBooleanValue= 1 
                else:
                    premiseBooleanValue = 0
                
                if conclusionCounter > 0:
                    conclusionBooleanValue= 1 
                else:
                    conclusionBooleanValue= 0
                
                self.featureArray.append([premiseBooleanValue, conclusionBooleanValue])
                
            elif self.featureSetConfiguration == 3:
                # [number of premise keywordsmodel in current sentence + number of conclusion keywordsmodel in current sentence]
                self.featureArray.append([premiseCounter + conclusionCounter])
                
            elif self.featureSetConfiguration == 4:
                # [boolean indicating if previous sentence contained some keyword or not, number of premise keywordsmodel in current sentence, number of conclusion keywordsmodel in current sentence]
                #TODO: is this feature relevant?
                
                
                if int(sentenceTuple[1]) == 0:
                    self.featureArray.append([0, premiseCounter, conclusionCounter])
                else:
                    # previous sentence in article info retrieved from database
                    previousSentence= sentenceCollection.find_one({"$and":[{"articleId": sentenceTuple[0]}, {"sentenceId": int(sentenceTuple[1]) - 1}]})
                    
                    premiseCounterPreviousSentence= 0
                    conclusionCounterPreviousSentence= 0
                    
                    
                    for tok in previousSentence["tokens"]:
                        
                        # argumentative keywords counters
                        if tok["lemma"] in self.premisseKeywordsList:
                            premiseCounterPreviousSentence= premiseCounterPreviousSentence + 1
                        
                        if tok["lemma"] in self.conclusionKeywordsList:
                            conclusionCounterPreviousSentence= conclusionCounterPreviousSentence + 1
                    
                    
                    self.featureArray.append([premiseCounterPreviousSentence + conclusionCounterPreviousSentence, premiseCounter, conclusionCounter])
                
            elif self.featureSetConfiguration == 5:
                # [boolean indicating if previous sentence contained some keyword or not, number of premise keywordsmodel in current sentence + number of conclusion keywordsmodel in current sentence]
                
                if int(sentenceTuple[1]) == 0:
                    self.featureArray.append([0, premiseCounter + conclusionCounter])
                else:
                    # previous sentence in article info retrieved from database
                    previousSentence= sentenceCollection.find_one({"$and":[{"articleId": sentenceTuple[0]}, {"sentenceId": int(sentenceTuple[1]) - 1}]})
                    
                    premiseCounterPreviousSentence= 0
                    conclusionCounterPreviousSentence= 0
                    
                    
                    for tok in previousSentence["tokens"]:
                        
                        # argumentative keywords counters
                        if tok["lemma"] in self.premisseKeywordsList:
                            premiseCounterPreviousSentence= premiseCounterPreviousSentence + 1
                        
                        if tok["lemma"] in self.conclusionKeywordsList:
                            conclusionCounterPreviousSentence= conclusionCounterPreviousSentence + 1
                    
                    
                    self.featureArray.append([premiseCounterPreviousSentence + conclusionCounterPreviousSentence, premiseCounter + conclusionCounter])
                
            elif self.featureSetConfiguration == 6:
                # [boolean indicating if previous sentence contained some keyword or not, boolean indicating if current sentence contains premise keyword, boolean indicating if current sentence contains conclusion keyword]
                
                previousSentenceCounter = 0
                
                if not (int(sentenceTuple[1]) == 0):
                    
                    # previous sentence in article info retrieved from database
                    previousSentence= sentenceCollection.find_one({"$and":[{"articleId": sentenceTuple[0]}, {"sentenceId": int(sentenceTuple[1]) - 1}]})
                    
                    premiseCounterPreviousSentence= 0
                    conclusionCounterPreviousSentence= 0
                    
                    
                    for tok in previousSentence["tokens"]:
                        
                        # argumentative keywords counters
                        if tok["lemma"] in self.premisseKeywordsList:
                            premiseCounterPreviousSentence= premiseCounterPreviousSentence + 1
                        
                        if tok["lemma"] in self.conclusionKeywordsList:
                            conclusionCounterPreviousSentence= conclusionCounterPreviousSentence + 1
                    
                    previousSentenceCounter= premiseCounterPreviousSentence + conclusionCounterPreviousSentence
                
                
                if previousSentenceCounter > 0:
                    if premiseCounter > 0:
                        if conclusionCounter > 0:
                            self.featureArray.append([1,1,1])
                        else:
                            self.featureArray.append([1,1,0])
                    else:
                        if conclusionCounter > 0:
                            self.featureArray.append([1,0,1])
                        else:
                            self.featureArray.append([1,0,0])
                else:
                    if premiseCounter > 0:
                        if conclusionCounter > 0:
                            self.featureArray.append([0,1,1])
                        else:
                            self.featureArray.append([0,1,0])
                    else:
                        if conclusionCounter > 0:
                            self.featureArray.append([0,0,1])
                        else:
                            self.featureArray.append([0,0,0])
            
            elif self.featureSetConfiguration == 7:
                # [boolean indicating if previous sentence contained some keyword or not, boolean indicating if current sentence contains premise keyword OR contains conclusion keyword]
                
                previousSentenceCounter = 0
                
                if not (int(sentenceTuple[1]) == 0):
                    
                    # previous sentence in article info retrieved from database
                    previousSentence= sentenceCollection.find_one({"$and":[{"articleId": sentenceTuple[0]}, {"sentenceId": int(sentenceTuple[1]) - 1}]})
                    
                    premiseCounterPreviousSentence= 0
                    conclusionCounterPreviousSentence= 0
                    
                    
                    for tok in previousSentence["tokens"]:
                        
                        # argumentative keywords counters
                        if tok["lemma"] in self.premisseKeywordsList:
                            premiseCounterPreviousSentence= premiseCounterPreviousSentence + 1
                        
                        if tok["lemma"] in self.conclusionKeywordsList:
                            conclusionCounterPreviousSentence= conclusionCounterPreviousSentence + 1
                    
                    previousSentenceCounter= premiseCounterPreviousSentence + conclusionCounterPreviousSentence
                
                
                if previousSentenceCounter > 0:
                    if (premiseCounter > 0) or (conclusionCounter > 0):
                        self.featureArray.append([1,1])
                    else:
                        self.featureArray.append([1,0])
                else:
                    if (premiseCounter > 0) or (conclusionCounter > 0):
                        self.featureArray.append([0,1])
                    else:
                        self.featureArray.append([0,0])
        # close connection
        mongoClient.close()
        
        return np.asarray(self.featureArray)

    def fit(self, X, y=None, **fit_params):
        return self
    
    def get_feature_names(self):
        if self.featureSetConfiguration == 0:
            return ['featureNotActive']
        elif self.featureSetConfiguration == 1:
            return ['keywordBoolean']
        elif self.featureSetConfiguration == 2:
            return ['premiseCounter', 'conclusionCounter']
        elif self.featureSetConfiguration == 3:
            return ['premiseCounter + conclusionCounter']
        elif self.featureSetConfiguration == 4:
            return ['lastCount', 'premiseCounter', 'conclusionCounter']
        elif self.featureSetConfiguration == 5:
            return ['lastCount', 'premiseCounter + conclusionCounter']
        elif self.featureSetConfiguration == 6:
            return ['lastBoolean', 'premiseBoolean', 'conclusionBoolean']
        elif self.featureSetConfiguration == 7:
            return ['lastBoolean', 'premiseBoolean or conclusionBoolean']
        elif self.featureSetConfiguration == 8:
            return self.argKeywordsList
    
    def get_content(self):
        return np.asarray(self.featureArray)
