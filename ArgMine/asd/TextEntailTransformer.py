"""
TextEntailTransformer:
"""

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from pymongo import MongoClient




class TextEntailTransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, config= 3):
        
        self.config= config
        
    
    def transform(self, X, **transform_params):
        
        self.featureArray= []
        
        if (self.config == 0):
            self.featureArray= [[0] for j in xrange(len(X))]
        else:
            
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            
            dbArgMine = mongoClient.ArgMineCorpus
            
            # Sentence's table
            textualEntailmentPredictionCollection= dbArgMine.textualEntailmentPrediction
            
            
            for learningInstance in X:
                
                
                # Feature Set for current learning instance
                currentLearningInstanceFeatureSet= []
                
                # get Textual Entailment Predictions for current sentence -> only returning the projections field
                currentSentenceEntailmentPredictions= textualEntailmentPredictionCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": learningInstance[1]}]}, {"_id": 0, "predictions": 1})
                
                currentSentencePredictions = currentSentenceEntailmentPredictions["predictions"] #[prediction["predictions"] for prediction in currentSentenceEntailmentPredictions]
                
                
                # Binary configuration
                if self.config == 1:
                    
                    if ("none" in [prediction["prediction"] for prediction in currentSentencePredictions]):
                        currentLearningInstanceFeatureSet.append(0)
                    else:
                        currentLearningInstanceFeatureSet.append(1)
                    
                elif self.config == 2:
                    
                    # determine prediction with highest entailment predictions ("entailment" or "paraphrase")
                    bestEntailmentPredictionScore= 0.0
                    bestEntailmentPredictionTuple= None
                    
                    for prediction in currentSentencePredictions:
                        if prediction["probaPredictions"]["entailment"] > bestEntailmentPredictionScore:
                            bestEntailmentPredictionTuple= prediction["probaPredictions"]
                            bestEntailmentPredictionScore= prediction["probaPredictions"]["entailment"]
                        
                        if prediction["probaPredictions"]["paraphrase"] > bestEntailmentPredictionScore:
                            bestEntailmentPredictionTuple= prediction["probaPredictions"]
                            bestEntailmentPredictionScore= prediction["probaPredictions"]["paraphrase"]
                        
                    
                    # add scores to feature set
                    if bestEntailmentPredictionTuple is None:
                        # if the article has only one sentence
                        currentLearningInstanceFeatureSet.append(0)
                        currentLearningInstanceFeatureSet.append(0)
                    else:
                        currentLearningInstanceFeatureSet.append(bestEntailmentPredictionTuple["none"])
                        
                        if bestEntailmentPredictionTuple["entailment"] > bestEntailmentPredictionTuple["paraphrase"]:
                            currentLearningInstanceFeatureSet.append(bestEntailmentPredictionTuple["entailment"])
                        else:
                            currentLearningInstanceFeatureSet.append(bestEntailmentPredictionTuple["paraphrase"])
                    
                    
                elif self.config == 3:
                    
                    # determine prediction with highest entailment predictions ("entailment" or "paraphrase")
                    bestEntailmentPredictionScore= 0.0
                    bestEntailmentPredictionTuple= None
                    
                    
                    for prediction in currentSentencePredictions:
                        if prediction["probaPredictions"]["entailment"] > bestEntailmentPredictionScore:
                            bestEntailmentPredictionTuple= prediction["probaPredictions"]
                            bestEntailmentPredictionScore= prediction["probaPredictions"]["entailment"]
                        
                        if prediction["probaPredictions"]["paraphrase"] > bestEntailmentPredictionScore:
                            bestEntailmentPredictionTuple= prediction["probaPredictions"]
                            bestEntailmentPredictionScore= prediction["probaPredictions"]["paraphrase"]
                        
                    
                    
                    # add scores to feature set
                    
                    if bestEntailmentPredictionTuple is None:
                        # if the article has only one sentence
                        currentLearningInstanceFeatureSet.append(0)
                    else:
                        if bestEntailmentPredictionTuple["entailment"] > bestEntailmentPredictionTuple["paraphrase"]:
                            currentLearningInstanceFeatureSet.append(bestEntailmentPredictionTuple["entailment"])
                        else:
                            currentLearningInstanceFeatureSet.append(bestEntailmentPredictionTuple["paraphrase"])
                    
                    
                
                
                
                # add Feature Set for current learning instance
                self.featureArray.append(currentLearningInstanceFeatureSet)
            
            # close connection
            mongoClient.close()
            
        
        return np.asarray(self.featureArray)
        
        
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    
    def get_feature_names(self):
        
        textEntailFeaturesNames= []
        
        if (self.config == 0):
            return ['None']
        elif (self.config == 1):
            textEntailFeaturesNames.append('PropositionInEntailRelation')
            
        elif (self.config == 2):
            textEntailFeaturesNames.append('PropositionNoneRelationConfidence')
            textEntailFeaturesNames.append('PropositionEntailmentRelationConfidence')
        elif (self.config == 3):
            textEntailFeaturesNames.append('PropositionInEntailRelationConfidenceScore')
        
        return textEntailFeaturesNames
    
    def get_content(self):
        return np.asarray(self.featureArray)
    
    
    
    