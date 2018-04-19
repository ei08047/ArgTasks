"""
SemanticLevelTransformer:
"""
from __future__ import division

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

import numpy as np
import time


class SemanticTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, wordnet, wordEmbeddingsModel, cosineSimilarity= 1, currentPropositionVersor= 1):
        
        self.wordnet= wordnet
        
        self.wordEmbeddingsModel= wordEmbeddingsModel
        
        self.cosineSimilarity= cosineSimilarity
        
        self.currentPropositionVersor= currentPropositionVersor
        
        #self.semanticDiff= semanticDiff

    def transform(self, X, **transform_params):
        
        self.featureArray= []
        
        
        if (self.cosineSimilarity == 0) and (self.currentPropositionVersor == 0):
            self.featureArray= [[0] for j in xrange(len(X))]
        else:
            
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            
            dbArgMine = mongoClient.ArgMineCorpus
            
            # Sentence's table
            sentenceCollection= dbArgMine.sentence
            
            # noun, verb, adjective, number
            posTagMainWordsList= ['N', 'A', 'Z', 'V']
            
            
            for learningInstance in X:
                
                # current learning instance info from database
                currentSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": learningInstance[1]}]})
                
                # Feature Set for current learning instance
                currentLearningInstanceFeatureSet= []
                
                
                # word embeddings model
                # cosine similarity
                if (self.cosineSimilarity == 1):
                    
                    currentPropositionTokens= [tok["lemma"] for tok in currentSentence["tokens"] if (tok["tags"][0] in posTagMainWordsList) and (not (tok["tags"][0:2] == "NP"))]
                    
                    previousPropositionTokens= []
                    
                    # previous Proposition
                    if not (int(currentSentence["sentenceId"]) == 0):
                        # previous sentence from database
                        previousSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": int(learningInstance[1]) - 1}]})
                        previousPropositionTokens= [tok["lemma"] for tok in previousSentence["tokens"] if (tok["tags"][0] in posTagMainWordsList) and (not (tok["tags"][0:2] == "NP"))]
                    
                    nextPropositionTokens= []
                    
                    # next sentence from database
                    nextSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": int(learningInstance[1]) + 1}]})
                    
                    if nextSentence is not None:
                        nextPropositionTokens= [tok["lemma"] for tok in nextSentence["tokens"] if (tok["tags"][0] in posTagMainWordsList) and (not (tok["tags"][0:2] == "NP"))]
                    
                    # get vector for Current Proposition
                    currentPropositionVector= self.getSentenceRepresentationFromWordEmbeddingsModel(currentPropositionTokens)
                    
                    # get vector for Previous Proposition
                    previousPropositionVector= self.getSentenceRepresentationFromWordEmbeddingsModel(previousPropositionTokens)
                    
                    currentLearningInstanceFeatureSet.append(self.vectorsSimilarityMetric(currentPropositionVector, previousPropositionVector))
                    
                    
                    
                    # get vector for Next Proposition
                    nextPropositionVector= self.getSentenceRepresentationFromWordEmbeddingsModel(nextPropositionTokens)
                    
                    currentLearningInstanceFeatureSet.append(self.vectorsSimilarityMetric(currentPropositionVector, nextPropositionVector))
                    
                    
                    
                
                
                if (self.currentPropositionVersor == 1):
                    
                    currentPropositionTokens= [tok["lemma"] for tok in currentSentence["tokens"] if (tok["tags"][0] in posTagMainWordsList) and (not (tok["tags"][0:2] == "NP"))]
                    
                    # get vector for Current Proposition
                    currentPropositionVector= self.getSentenceRepresentationFromWordEmbeddingsModel(currentPropositionTokens)
                    
                    
                    
                    # calculate cosine similarity between re1Vector and re2Vector
                    if (currentPropositionVector is None):
                        # get number of dimensions in embeddings space
                        #WARNING: this was the only way I found to retrieve the number of dimensions of word in the embedding space
                        #TODO: Is there a better way to compute this?
                        numberOfDimensions= (self.wordEmbeddingsModel).get('azul').shape[0]
                        
                        currentLearningInstanceFeatureSet= currentLearningInstanceFeatureSet + [0 for j in xrange(numberOfDimensions)]
                        
                    else:
                        currentLearningInstanceFeatureSet = currentLearningInstanceFeatureSet + currentPropositionVector.tolist()
                    
                    
                
                
                # add Feature Set for current learning instance
                self.featureArray.append(currentLearningInstanceFeatureSet)
            
            # close connection
            mongoClient.close()
            
        
        
        return np.asarray(self.featureArray)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        
        semanticLevelFeaturesNames= []
        
        if (self.cosineSimilarity == 0) and (self.currentPropositionVersor == 0):
            return ['None']
        
        if (self.cosineSimilarity > 0):
            semanticLevelFeaturesNames.append('CosineSimilarityScore_Current_Previous')
            semanticLevelFeaturesNames.append('CosineSimilarityScore_Current_Next')
        
        if self.currentPropositionVersor > 0:
            #WARNING this was the only way I found to retrieve the number of dimensions of word in the embedding space
            #TODO Is there a better way to compute this?
            numberOfDimensions= (self.wordEmbeddingsModel).get('azul').shape[0]
            
            for i in xrange(numberOfDimensions):
                semanticLevelFeaturesNames.append('CurrentPropoistionVersor_' + str(i))
        
        
        return semanticLevelFeaturesNames

    def get_content(self):
        return np.asarray(self.featureArray)

    def getSentenceRepresentationFromWordEmbeddingsModel(self, words):
        
        firstTokenIndexOnEmbeddingsModel= 0
        currentPropositionNumberOfTokensInVecSum= 0
        
        currentPropositionVector= None
        
        for tokenIndex in xrange(len(words)):
            tokenVector= (self.wordEmbeddingsModel).get(words[tokenIndex])
            if tokenVector is not None:
                currentPropositionVector= tokenVector
                firstTokenIndexOnEmbeddingsModel= tokenIndex
                currentPropositionNumberOfTokensInVecSum += 1
            
        
        if currentPropositionVector is not None:
            for tokenIndex in xrange(firstTokenIndexOnEmbeddingsModel + 1, len(words)):
                
                currentTokenVector= (self.wordEmbeddingsModel).get(words[tokenIndex])
                
                if currentTokenVector is not None:
                    currentPropositionVector= np.add(currentPropositionVector, currentTokenVector)
                    currentPropositionNumberOfTokensInVecSum += 1
        
        # normalize text vector
        if currentPropositionVector is not None:
            currentPropositionVector= np.true_divide(currentPropositionVector, currentPropositionNumberOfTokensInVecSum)
        
        return currentPropositionVector

    def vectorsSimilarityMetric(self, vector1, vector2):
        #TODO: Currently only the cosine similarity metric is implemented. Explore other metrics?
        
        # calculate cosine similarity between Current Proposition Vector and Previous Proposition Vector
        if (vector1 is None) or (vector2 is None):
            return 1.0
        else:
            
            sklearnCosineSimilarity= cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))
            
            if abs(sklearnCosineSimilarity[0][0] - 1.0) < 0.01:
                return 0.0
            else:
                return float((1.0 - sklearnCosineSimilarity[0][0]) / 2.0)