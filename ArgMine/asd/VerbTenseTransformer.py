"""
VerbTenseTransformer:
"""

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from pymongo import MongoClient


class VerbTenseTransformer(TransformerMixin, BaseEstimator):
    
    
    def __init__(self, featureSetConfiguration= 3):
        self.featureSetConfiguration= featureSetConfiguration

    def transform(self, X, **transform_params):
        
        self.featureArray= []
        
        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        
        dbArgMine = mongoClient.ArgMineCorpus
        
        # Sentence's table
        sentenceCollection= dbArgMine.sentence
        
        for learningInstance in X:
            
            # current learning instance info from database
            currentSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": learningInstance[1]}]})
            
            mainVerbsTagsList= []
            
            # get tags of all main verbs from current sentence
            for t in currentSentence["tokens"]:
                
                if t["tags"][0] == 'V':
                    # current token is a verb
                    if t["tags"][1] == 'M':
                        # current token is main verb
                        mainVerbsTagsList.append(t["tags"])
            
            if self.featureSetConfiguration == 0:
                # boolean feature indication if there is change in tense verb or not inside current proposition
                verbTenseChange= False
                
                for verbIndex in xrange(1, len(mainVerbsTagsList)):
                    if (not (mainVerbsTagsList[verbIndex][3] == mainVerbsTagsList[verbIndex - 1][3])) and ( not (mainVerbsTagsList[verbIndex][3] == '0')) and (not (mainVerbsTagsList[verbIndex - 1][3] == '0')):
                        verbTenseChange= True
                        break
                
                if verbTenseChange:
                    self.featureArray.append([1])
                else:
                    self.featureArray.append([0])
                
            elif self.featureSetConfiguration == 1:
                # boolean feature indication if there is change in tense verb or not inside current proposition
                # reduced tense verb class set: None(0), Past(1), Present(2), Future(3)
                verbTenseChange= False
                
                mainVerbsReducedTagsList = []
                
                for verbIndex in xrange(len(mainVerbsTagsList)):
                    if mainVerbsTagsList[verbIndex][3] == '0':
                        mainVerbsReducedTagsList.append(0)
                    elif mainVerbsTagsList[verbIndex][3] == 'P':
                        mainVerbsReducedTagsList.append(2)
                    elif mainVerbsTagsList[verbIndex][3] == 'I':
                        mainVerbsReducedTagsList.append(1)
                    elif mainVerbsTagsList[verbIndex][3] == 'F':
                        mainVerbsReducedTagsList.append(3)
                    elif mainVerbsTagsList[verbIndex][3] == 'S':
                        mainVerbsReducedTagsList.append(1)
                    elif mainVerbsTagsList[verbIndex][3] == 'C':
                        mainVerbsReducedTagsList.append(1)
                
                for verbIndex in xrange(1, len(mainVerbsTagsList)):
                    if (not (mainVerbsReducedTagsList[verbIndex] == mainVerbsTagsList[verbIndex - 1])) and ( not (mainVerbsTagsList[verbIndex] == 0)) and (not (mainVerbsTagsList[verbIndex - 1] == 0)):
                        verbTenseChange= True
                        break
                
                if verbTenseChange:
                    self.featureArray.append([1])
                else:
                    self.featureArray.append([0])
                
                
                
            elif self.featureSetConfiguration == 2:
                
                currentPropositionFeatureSet= []
                
                # verb tenses in current sentence
                currentPropositionVerbTensesInfo= getVerbTensesStatsInSentence(currentSentence["tokens"])
                
                
                # current proposition verb tense changes
                if len([elem for elem in currentPropositionVerbTensesInfo if elem > 0]) > 0:
                    currentPropositionFeatureSet.append(1)
                else:
                    currentPropositionFeatureSet.append(0)
                
                
                # current proposition and previous proposition verb tense change
                if int(currentSentence["sentenceId"]) == 0:
                    currentPropositionFeatureSet.append(0)
                else:
                    
                    # previous sentence from database
                    previousSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": int(learningInstance[1]) - 1}]})
                    
                    # verb tenses in previous sentence
                    previousPropositionVerbTensesInfo= getVerbTensesStatsInSentence(previousSentence["tokens"])
                    
                    verbTenseChange= False
                    for verbTenseIndex in xrange(1, len(currentPropositionVerbTensesInfo)):
                        if not ( (currentPropositionVerbTensesInfo[verbTenseIndex] > 0) and (previousPropositionVerbTensesInfo[verbTenseIndex] > 0)):
                            verbTenseChange= True
                            break
                        
                    
                    if verbTenseChange:
                        currentPropositionFeatureSet.append(1)
                    else:
                        currentPropositionFeatureSet.append(0)
                    
                    
                
                # next sentence from database
                nextSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": int(learningInstance[1]) + 1}]})
                
                # current proposition and next proposition verb tense change
                if nextSentence is None:
                    currentPropositionFeatureSet.append(0)
                else:
                    
                    # verb tenses in next sentence
                    nextPropositionVerbTensesInfo= getVerbTensesStatsInSentence(nextSentence["tokens"])
                    
                    verbTenseChange= False
                    for verbTenseIndex in xrange(1, len(currentPropositionVerbTensesInfo)):
                        if not ( (currentPropositionVerbTensesInfo[verbTenseIndex] > 0) and (nextPropositionVerbTensesInfo[verbTenseIndex] > 0)):
                            verbTenseChange= True
                            break
                        
                    
                    if verbTenseChange:
                        currentPropositionFeatureSet.append(1)
                    else:
                        currentPropositionFeatureSet.append(0)
                    
                    
                
                self.featureArray.append(currentPropositionFeatureSet)
                
                
                
                
            elif self.featureSetConfiguration == 3:
                
                currentPropositionFeatureSet= []
                
                # current proposition verb tense changes
                
                # verb tenses in current sentence
                currentPropositionVerbTensesInfo= getVerbTensesStatsInSentence(currentSentence["tokens"])
                
                
                # current proposition verb tense changes
                if len([elem for elem in currentPropositionVerbTensesInfo if elem > 0]) > 0:
                    currentPropositionFeatureSet.append(1)
                else:
                    currentPropositionFeatureSet.append(0)
                
                
                
                # current proposition and previous proposition verb tense changes
                
                # check if last seen main verb from previous proposition has the same tense verb as first main verb
                # occurring in the current proposition
                
                if int(currentSentence["sentenceId"]) == 0:
                    currentPropositionFeatureSet.append(0)
                else:
                    
                    currentSentenceTokens= currentSentence["tokens"]
                    
                    currentSentenceFirstMainVerb= None
                    
                    for tokenIndex in xrange(len(currentSentenceTokens)):
                        if currentSentenceTokens[tokenIndex]["tags"][0] == 'V':
                            # current token is a verb
                            if currentSentenceTokens[tokenIndex]["tags"][1] == 'M':
                                # current token is main verb
                                currentSentenceFirstMainVerb= currentSentenceTokens[tokenIndex]
                                break
                    
                    # previous sentence from database
                    previousSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": int(learningInstance[1]) - 1}]})
                    
                    previousSentenceTokens= previousSentence["tokens"]
                    
                    previousSentenceLastMainVerb= None
                    
                    for tokenIndex in xrange(len(previousSentenceTokens)):
                        if previousSentenceTokens[tokenIndex]["tags"][0] == 'V':
                            # current token is a verb
                            if previousSentenceTokens[tokenIndex]["tags"][1] == 'M':
                                # current token is main verb
                                previousSentenceLastMainVerb= previousSentenceTokens[tokenIndex]
                    
                    
                    if (currentSentenceFirstMainVerb is None) or (previousSentenceLastMainVerb is None):
                        currentPropositionFeatureSet.append(0)
                    else:
                        if currentSentenceFirstMainVerb["tags"][3] == previousSentenceLastMainVerb["tags"][3]:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                    
                    
                
                # next sentence from database
                nextSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": int(learningInstance[1]) + 1}]})
                
                if nextSentence is None:
                    currentPropositionFeatureSet.append(0)
                else:
                    
                    currentSentenceTokens= currentSentence["tokens"]
                    
                    currentSentenceLastMainVerb= None
                    
                    for tokenIndex in xrange(len(currentSentenceTokens)):
                        if currentSentenceTokens[tokenIndex]["tags"][0] == 'V':
                            # current token is a verb
                            if currentSentenceTokens[tokenIndex]["tags"][1] == 'M':
                                # current token is main verb
                                currentSentenceLastMainVerb= currentSentenceTokens[tokenIndex]
                    
                    
                    nextSentenceTokens= nextSentence["tokens"]
                    
                    nextSentenceFirstMainVerb= None
                    
                    for tokenIndex in xrange(len(nextSentenceTokens)):
                        if nextSentenceTokens[tokenIndex]["tags"][0] == 'V':
                            # current token is a verb
                            if nextSentenceTokens[tokenIndex]["tags"][1] == 'M':
                                # current token is main verb
                                nextSentenceFirstMainVerb= nextSentenceTokens[tokenIndex]
                    
                    
                    if (currentSentenceLastMainVerb is None) or (nextSentenceFirstMainVerb is None):
                        currentPropositionFeatureSet.append(0)
                    else:
                        if currentSentenceLastMainVerb["tags"][3] == nextSentenceFirstMainVerb["tags"][3]:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                    
                    
                
                
                
                self.featureArray.append(currentPropositionFeatureSet)
                
                
            elif self.featureSetConfiguration == 4:
                self.featureArray.append([0])
        
        
        # close connection
        mongoClient.close()
        
        return np.asarray(self.featureArray)

    def fit(self, X, y=None, **fit_params):
        return self
    
    def get_feature_names(self):
        if (self.featureSetConfiguration == 0) or (self.featureSetConfiguration == 1):
            return ['changeInVerbTense']
        elif (self.featureSetConfiguration == 2) or (self.featureSetConfiguration == 3):
            return ['changeInVerbTenseInsideCurrentSentence', 'changeInVerbTenseCurrentPreviousSentence', 'changeInVerbTenseCurrentFollowingSentence']
        elif self.featureSetConfiguration == 4:
            return ['None']

    def get_content(self):
        return np.asarray(self.featureArray)


def getVerbTensesStatsInSentence(tokens):
    # Verb Tense taxonomy
    # 0 -> '0'
    # 1 -> 'S'
    # 2 -> 'P'
    # 3 -> 'F'
    # 4 -> 'I'
    # 5 -> 'C' 
    verbTenses = [0,0,0,0,0,0]
    
    
    for tokenIndex in range(len(tokens)):
        if tokens[tokenIndex]["tags"][0] == 'V':
            # current token is a verb
            if tokens[tokenIndex]["tags"][1] == 'M':
                # current token is main verb
                #mainVerbsTagsList.append(t.tags)
                
                # Since the distinction between main verb and auxiliary is not working in the POSTagger
                # we have to use this heuristic to avoid auxiliary verbs
                if (tokenIndex > 0) and (not tokens[tokenIndex - 1]["tags"][0] == 'V'):
                    
                    if tokens[tokenIndex]["tags"][3] == '0':
                        verbTenses[0] = verbTenses[0] + 1
                    elif tokens[tokenIndex]["tags"][3] == 'P':
                        verbTenses[1] = verbTenses[1] + 1
                    elif tokens[tokenIndex]["tags"][3] == 'I':
                        verbTenses[2] = verbTenses[2] + 1
                    elif tokens[tokenIndex]["tags"][3] == 'F':
                        verbTenses[3] = verbTenses[3] + 1
                    elif tokens[tokenIndex]["tags"][3] == 'S':
                        verbTenses[4] = verbTenses[4] + 1
                    elif tokens[tokenIndex]["tags"][3] == 'C':
                        verbTenses[5] = verbTenses[5] + 1
                    else:
                        verbTenses[0] = verbTenses[0] + 1
                    
                
                
    
    return verbTenses

