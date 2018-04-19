"""
TextStatisticsTransformer: Custom Transformer with the purpose to create the Word Couples features.
It receives as input a matrix with dimension X where each element is a sentence/proposition
(being X the number of propositions in the dataset). It outputs a matrix with dimension 
X:Y, where the value of Y depends on the value of each of the following parameters:
'sentenceLength', 'averageWordLength', 'punctuationMarks' and 'absolutePosition'
"""

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from pymongo import MongoClient


class TextStatisticsTransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, sentenceLength= False, averageWordLength= False, punctuationMarks= False, absolutePosition= True):
        
        self.sentenceLength= sentenceLength
        self.averageWordLength= averageWordLength
        self.punctuationMarks= punctuationMarks
        self.absolutePosition= absolutePosition

    def transform(self, X, **transform_params):
        # X -> corresponds to the data -> array of propositions
        self.featureArray= []
        if (not self.sentenceLength) and (not self.averageWordLength) and (not self.punctuationMarks) and (not self.absolutePosition):
            self.featureArray= [[0] for j in range(len(X))]
        else:
            
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            
            dbArgMine = mongoClient.ArgMineCorpus
            
            # Sentence's table
            sentenceCollection= dbArgMine.sentence
            
            for learningInstance in X:
                
                currentPropositionFeatureArray= []
                
                # current learning instance info from database
                currentSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": learningInstance[1]}]})
                
                if self.sentenceLength:
                    currentPropositionFeatureArray.append(len(currentSentence["tokens"]))
                
                if self.averageWordLength:
                    
                    if (len(currentSentence["tokens"]) == 0):
                        currentPropositionFeatureArray.append(0.0)
                    else:
                        tokensLengthAcumulator= 0
                        
                        for tok in currentSentence["tokens"]:
                            #TODO: LEMMAS VS ORIGINAL CONTENT
                            tokensLengthAcumulator += len(tok["content"])
                        
                        currentPropositionFeatureArray.append(tokensLengthAcumulator / float(len(currentSentence["tokens"])))
                
                if self.punctuationMarks:
                    
                    currentPropositionFeatureArray.append(len([tok["content"] for tok in currentSentence["tokens"] if tok["tags"][0] == 'F']))
                    
                
                if self.absolutePosition:
                    sentencesInArticleOrderedByArticleId= sentenceCollection.find({"articleId": learningInstance[0]}, {"sentenceId": 1, "_id": 0}).sort([("sentenceId", 1)])
                    sentencesIdsInArticle= [sentence["sentenceId"] for sentence in sentencesInArticleOrderedByArticleId]
                    
                    currentSentenceAbsolutePosition = 0
                    
                    for sentenceId in sentencesIdsInArticle:
                        if sentenceId == currentSentence["articleId"]:
                            break
                        currentSentenceAbsolutePosition += 1
                    
                    if len(sentencesIdsInArticle) == 0:
                        print ("\n\n[Warning] Number of sentences in Article is Zero in TextStatisticsTransformer!")
                        currentPropositionFeatureArray.append(0.0)
                    elif len(sentencesIdsInArticle) == 1:
                        currentPropositionFeatureArray.append(0.0)
                    else:
                        currentPropositionFeatureArray.append(float(currentSentenceAbsolutePosition) / float(len(sentencesIdsInArticle)))
                    
                
                
                (self.featureArray).append(currentPropositionFeatureArray)
            
            # close connection
            mongoClient.close()
        
        
        return np.asarray(self.featureArray)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def get_feature_names(self):
        featureNames= []
        if (not self.sentenceLength) and (not self.averageWordLength) and (not self.punctuationMarks) and (not self.absolutePosition):
            featureNames.append("None")
        else:
            if self.sentenceLength:
                featureNames.append("sentenceLength")
            
            if self.averageWordLength:
                featureNames.append("averageWordLength")
                
            if self.punctuationMarks:
                featureNames.append("punctuationMarks")
            
            if self.absolutePosition:
                featureNames.append("absolutePosition")
        
        return featureNames
    
    def get_content(self):
        return np.asarray(self.featureArray)
    