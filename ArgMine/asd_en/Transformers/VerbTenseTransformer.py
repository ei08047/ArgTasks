"""
VerbTenseTransformer:
"""

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from pymongo import MongoClient

#TODO: Several assumptions are made here! This Transformer should be improved in several ways.
class VerbTenseTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, featureSetConfiguration= 4):
        self.featureSetConfiguration= featureSetConfiguration

    def transform(self, X, **transform_params):
        
        self.featureArray= []
        
        if self.featureSetConfiguration == 0:
            self.featureArray= [[0] for j in range(len(X))]
        else:
            
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            
            aaecCorpus = mongoClient.AAECCorpus
            
            # Sentence's table
            sentenceCollection= aaecCorpus.sentence
            
            for learningInstance in X:
                
                currentLearningInstanceFeatureSet= []
                
                # current learning instance info from database
                currentSentence= sentenceCollection.find_one({"$and":[{"articleId": int(learningInstance[0])}, {"sentenceId": int(learningInstance[1])}]})
                
                mainVerbsTagsDict= getVerbTensesStatsInSentence(currentSentence["tokens"])
                
                # 1 -> Past -> VBD, VBN
                # 2 -> Present -> VBP, VBZ, VBG 
                # 3 -> Future -> MD
                mainVerbsReducedTagsList = {
                    "Past": mainVerbsTagsDict["VBD"] + mainVerbsTagsDict["VBN"],
                    "Present": mainVerbsTagsDict["VBP"] + mainVerbsTagsDict["VBZ"] + mainVerbsTagsDict["VBG"],
                    "Future": mainVerbsTagsDict["MD"]
                    }
                
                if self.featureSetConfiguration == 1:
                    # boolean feature indication if there is change in tense verb or not inside current proposition
                    verbTenseChange= False
                    
                    for verbTense in [k for k in mainVerbsTagsDict.keys() if not (k == "VB")]:
                        for otherVerbTense in [k for k in mainVerbsTagsDict.keys() if not ((k == "VB") or (k == verbTense))]:
                            if (mainVerbsTagsDict[verbTense] > 0) and (mainVerbsTagsDict[otherVerbTense] > 0):
                                verbTenseChange= True
                                break
                    
                    if verbTenseChange:
                        currentLearningInstanceFeatureSet.append(1)
                    else:
                        currentLearningInstanceFeatureSet.append(0)
                    
                elif self.featureSetConfiguration == 2:
                    # boolean feature indication if there is change in tense verb or not inside current proposition
                    # reduced tense verb class set: None(0), Past(1), Present(2), Future(3)
                    verbTenseChange= False
                    
                    for verbTense in mainVerbsReducedTagsList.keys():
                        for otherVerbTense in [k for k in mainVerbsReducedTagsList.keys() if not (k == verbTense)]:
                            if (mainVerbsReducedTagsList[verbTense] > 0) and (mainVerbsReducedTagsList[otherVerbTense] > 0):
                                verbTenseChange= True
                                break
                    
                    if verbTenseChange:
                        currentLearningInstanceFeatureSet.append(1)
                    else:
                        currentLearningInstanceFeatureSet.append(0)

                elif self.featureSetConfiguration == 3:
                    # verb tenses in current sentence
                    currentSentenceVerbTenseChange= False
                    for verbTense in [k for k in mainVerbsTagsDict.keys() if not (k == "VB")]:
                        for otherVerbTense in [k for k in mainVerbsTagsDict.keys() if not ((k == "VB") or (k == verbTense))]:
                            if (mainVerbsTagsDict[verbTense] > 0) and (mainVerbsTagsDict[otherVerbTense] > 0):
                                currentSentenceVerbTenseChange= True
                                break
                    if currentSentenceVerbTenseChange:
                        currentLearningInstanceFeatureSet.append(1)
                    else:
                        currentLearningInstanceFeatureSet.append(0)
                    # current proposition and previous proposition verb tense change
                    if int(currentSentence["sentenceId"]) == 0:
                        currentLearningInstanceFeatureSet.append(0)
                    else:
                        # previous sentence from database
                        previousSentence= sentenceCollection.find_one({"$and":[{"articleId": int(learningInstance[0])}, {"sentenceId": int(learningInstance[1]) - 1}]})
                        # verb tenses in previous sentence
                        previousPropositionVerbTensesDict= getVerbTensesStatsInSentence(previousSentence["tokens"])
                        verbTenseChange= False
                        for verbTense in [k for k in mainVerbsTagsDict.keys() if not (k == "VB")]:
                            if ((mainVerbsTagsDict[verbTense] > 0) and (previousPropositionVerbTensesDict[verbTense] == 0)) or ((mainVerbsTagsDict[verbTense] == 0) and (previousPropositionVerbTensesDict[verbTense] > 0)):
                                currentSentenceVerbTenseChange= True
                                break
                        if verbTenseChange:
                            currentLearningInstanceFeatureSet.append(1)
                        else:
                            currentLearningInstanceFeatureSet.append(0)
                    # next sentence from database
                    nextSentence= sentenceCollection.find_one({"$and":[{"articleId": int(learningInstance[0])}, {"sentenceId": int(learningInstance[1]) + 1}]})
                    # current proposition and next proposition verb tense change
                    if nextSentence is None:
                        currentLearningInstanceFeatureSet.append(0)
                    else:
                        # verb tenses in next sentence
                        nextPropositionVerbTensesInfo= getVerbTensesStatsInSentence(nextSentence["tokens"])
                        verbTenseChange= False
                        for verbTense in [k for k in mainVerbsTagsDict.keys() if not (k == "VB")]:
                            if ((mainVerbsTagsDict[verbTense] > 0) and (nextPropositionVerbTensesInfo[verbTense] == 0)) or ((mainVerbsTagsDict[verbTense] == 0) and (nextPropositionVerbTensesInfo[verbTense] > 0)):
                                currentSentenceVerbTenseChange= True
                                break
                        if verbTenseChange:
                            currentLearningInstanceFeatureSet.append(1)
                        else:
                            currentLearningInstanceFeatureSet.append(0)

                elif self.featureSetConfiguration == 4:
                    # current proposition verb tense changes
                    # verb tenses in current sentence
                    currentSentenceVerbTenseChange= False
                    for verbTense in [k for k in mainVerbsTagsDict.keys() if not (k == "VB")]:
                        for otherVerbTense in [k for k in mainVerbsTagsDict.keys() if not ((k == "VB") or (k == verbTense))]:
                            if (mainVerbsTagsDict[verbTense] > 0) and (mainVerbsTagsDict[otherVerbTense] > 0):
                                currentSentenceVerbTenseChange= True
                                break
                    
                    if currentSentenceVerbTenseChange:
                        currentLearningInstanceFeatureSet.append(1)
                    else:
                        currentLearningInstanceFeatureSet.append(0)
                    
                    
                    
                    # current proposition and previous proposition verb tense changes
                    
                    # check if last seen main verb from previous proposition has the same tense verb as first main verb
                    # occurring in the current proposition
                    
                    if int(currentSentence["sentenceId"]) == 0:
                        currentLearningInstanceFeatureSet.append(0)
                    else:
                        
                        currentSentenceTokens= currentSentence["tokens"]
                        
                        currentSentenceFirstMainVerb= None
                        
                        for tokenIndex in range(len(currentSentenceTokens)):
                            if (currentSentenceTokens[tokenIndex]["tags"][0] == 'V') or (currentSentenceTokens[tokenIndex]["tags"][0] == 'M'):
                            # current token is  verb
                            
                                # heuristic to avoid auxiliary verbs
                                #TODO: is this correct?
                                if (tokenIndex == 0) or ((tokenIndex > 0) and (not currentSentenceTokens[tokenIndex - 1]["tags"][0] == 'V')):
                                    # current token is main verb
                                    
                                    currentSentenceFirstMainVerb= currentSentenceTokens[tokenIndex]
                                    break
                        
                        # previous sentence from database
                        previousSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": int(learningInstance[1]) - 1}]})
                        
                        previousSentenceTokens= previousSentence["tokens"]
                        
                        previousSentenceLastMainVerb= None
                        
                        for tokenIndex in range(len(previousSentenceTokens)):
                            if (previousSentenceTokens[tokenIndex]["tags"][0] == 'V') or (previousSentenceTokens[tokenIndex]["tags"][0] == 'M'):
                            # current token is  verb
                            
                                # heuristic to avoid auxiliary verbs
                                #TODO: is this correct?
                                if (tokenIndex == 0) or ((tokenIndex > 0) and (not previousSentenceTokens[tokenIndex - 1]["tags"][0] == 'V')):
                                    # current token is main verb
                                    
                                    previousSentenceLastMainVerb= previousSentenceTokens[tokenIndex]
                        
                        
                        if (currentSentenceFirstMainVerb is None) or (previousSentenceLastMainVerb is None):
                            currentLearningInstanceFeatureSet.append(0)
                        else:
                            if currentSentenceFirstMainVerb["tags"] == previousSentenceLastMainVerb["tags"]:
                                currentLearningInstanceFeatureSet.append(1)
                            else:
                                currentLearningInstanceFeatureSet.append(0)
                        
                        
                    
                    # next sentence from database
                    nextSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": int(learningInstance[1]) + 1}]})
                    
                    if nextSentence is None:
                        currentLearningInstanceFeatureSet.append(0)
                    else:
                        
                        currentSentenceTokens= currentSentence["tokens"]
                        
                        currentSentenceLastMainVerb= None
                        
                        for tokenIndex in range(len(currentSentenceTokens)):
                            if (currentSentenceTokens[tokenIndex]["tags"][0] == 'V') or (currentSentenceTokens[tokenIndex]["tags"][0] == 'M'):
                            # current token is  verb
                            
                                # heuristic to avoid auxiliary verbs
                                #TODO: is this correct?
                                if (tokenIndex == 0) or ((tokenIndex > 0) and (not currentSentenceTokens[tokenIndex - 1]["tags"][0] == 'V')):
                                    # current token is main verb
                                    
                                    currentSentenceLastMainVerb= currentSentenceTokens[tokenIndex]
                                    
                        
                        
                        nextSentenceTokens= nextSentence["tokens"]
                        
                        nextSentenceFirstMainVerb= None
                        
                        for tokenIndex in range(len(nextSentenceTokens)):
                            if (nextSentenceTokens[tokenIndex]["tags"][0] == 'V') or (nextSentenceTokens[tokenIndex]["tags"][0] == 'M'):
                            # current token is  verb
                            
                                # heuristic to avoid auxiliary verbs
                                #TODO: is this correct?
                                if (tokenIndex == 0) or ((tokenIndex > 0) and (not nextSentenceTokens[tokenIndex - 1]["tags"][0] == 'V')):
                                    # current token is main verb
                                    
                                    nextSentenceFirstMainVerb= nextSentenceTokens[tokenIndex]
                        
                        
                        if (currentSentenceLastMainVerb is None) or (nextSentenceFirstMainVerb is None):
                            currentLearningInstanceFeatureSet.append(0)
                        else:
                            if currentSentenceLastMainVerb["tags"][3] == nextSentenceFirstMainVerb["tags"][3]:
                                currentLearningInstanceFeatureSet.append(1)
                            else:
                                currentLearningInstanceFeatureSet.append(0)

                # add Feature Set for current learning instance
                self.featureArray.append(currentLearningInstanceFeatureSet)

            # close connection
            mongoClient.close()
        
        return np.asarray(self.featureArray)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def get_feature_names(self):
        if (self.featureSetConfiguration == 1) or (self.featureSetConfiguration == 2):
            return ['changeInVerbTense']
        elif (self.featureSetConfiguration == 3) or (self.featureSetConfiguration == 4):
            return ['changeInVerbTenseInsideCurrentSentence', 'changeInVerbTenseCurrentPreviousSentence', 'changeInVerbTenseCurrentFollowingSentence']
        elif self.featureSetConfiguration == 0:
            return ['None']

    def get_content(self):
        return np.asarray(self.featureArray)

def getVerbTensesStatsInSentence(tokens):
    # Verb Tense taxonomy
    # VB: base form
    # VBD: past tense
    # VBG: gerund or present participle
    # VBN: past participle
    # VBP: non-3rd person singular present
    # VBZ: 3rd person singular present
    # MD: Modal
    verbTenses = {"VB": 0, "VBD": 0, "VBG": 0, "VBN": 0, "VBP": 0, "VBZ": 0, "MD": 0}
    
    
    for tokenIndex in range(len(tokens)):
        if (tokens[tokenIndex]["tags"][0] == 'V') or (tokens[tokenIndex]["tags"][0] == 'M'):
            # current token is  verb
            
            # heuristic to avoid auxiliary verbs
            #TODO: is this correct?
            if (tokenIndex == 0) or ((tokenIndex > 0) and (not tokens[tokenIndex - 1]["tags"][0] == 'V')):
                
                currentTokenPoSTags= tokens[tokenIndex]["tags"]
                
                if currentTokenPoSTags in verbTenses:
                    verbTenses[currentTokenPoSTags]= verbTenses[currentTokenPoSTags] + 1
                else:
                    raise Exception("Parsing error: main verb PoS Tag not found in verb tenses list at VerbTenseTransformer.getVerbTensesStatsInSentence()")
                
            
            
    
    return verbTenses

