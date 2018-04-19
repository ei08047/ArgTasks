"""
RabbitRuleTransformer: Custom Transformer to create a feature set capturing the Rabbit Rule intuition. 
A Transformer is a sklearn object that should implement the following methods: fit, 
transform and fit_transform (inheriting from TransformerMixin implements this method for 
free).
Generally, they accept a matrix as input and return a matrix of the same shape as output ( 
with the corresponding features values for each input element).
In this particular case, it receives as input a matrix with dimensions 1:X where each element 
X is a proposition. It outputs a matrix with dimension X:Y where X corresponds to the total 
number of propositions received as input and Y depends on the value of the 
'featureSetConfiguration' parameter
"""

from sklearn.base import TransformerMixin, BaseEstimator

import numpy as np
import time

from pymongo import MongoClient


class RabbitRuleTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, wordnet, wordEmbeddingsModel, featureSetConfiguration= 1):
        self.featureSetConfiguration= featureSetConfiguration
        
        self.wordnet= wordnet
        
        self.wordEmbeddingsModel= wordEmbeddingsModel

    def transform(self, X, **transform_params):
        # X -> set of learning instances
        
        self.featureArray= []
        
        if (self.featureSetConfiguration == 0):
            self.featureArray= [[0] for j in range(len(X))]
        else:
            
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            aaecCorpus = mongoClient.AAECCorpus
            # Sentence's table
            sentenceCollection= aaecCorpus.sentence
            for learningInstance in X:
                
                # current learning instance info from database
                currentSentence= sentenceCollection.find_one({"$and":[{"articleId": int(learningInstance[0])}, {"sentenceId": int(learningInstance[1])}]})
                
                currentPropositionFeatureSet= []
                
                # adjective, modal, noun, verb, 
                posTagMainWordsList= ['JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                
                posTagNouns= ['NN', 'NNS']
                
                posTagProperNoun= ['NNP', 'NNPS']
                
                
                # current Proposition
                currentPropositionAdjectiveList= [t["lemma"] for t in currentSentence["tokens"] if (t["tags"][0] == 'J')]
                
                currentPropositionNounList= [t["lemma"] for t in currentSentence["tokens"] if (t["tags"] in posTagNouns)]
                
                currentPropositionVerbList= [t["lemma"] for t in currentSentence["tokens"] if (t["tags"][0] == 'V')]
                
                #TODO: Information regarding Coreferences would be very useful at this stage
                currentPropositionNamedEntityList= [t["lemma"] for t in currentSentence["tokens"] if (t["tags"] in posTagProperNoun)]
                
                currentPropositionMainTokens= [tok for tok in currentSentence["tokens"] if (tok["tags"] in posTagMainWordsList)]
                
                currentPropositionMainWords= list(set([tok["lemma"] for tok in currentPropositionMainTokens if not (tok["tags"] in posTagProperNoun)]))
                
                
                if self.featureSetConfiguration == 1:
                    
                    
                    # Check Adjective (A) repetitions inside current proposition
                    # Note: a list contains elements with repetitions but a set contains elements without repetitions
                    if len(set(currentPropositionAdjectiveList)) < len(currentPropositionAdjectiveList):
                        currentPropositionFeatureSet.append(1)
                    else:
                        currentPropositionFeatureSet.append(0)
                    
                    # Check Noun (N) repetitions inside current proposition
                    if len(set(currentPropositionNounList)) < len(currentPropositionNounList):
                        currentPropositionFeatureSet.append(1)
                    else:
                        currentPropositionFeatureSet.append(0)
                    
                    # Check Verb (V) repetitions inside current proposition
                    if len(set(currentPropositionVerbList)) < len(currentPropositionVerbList):
                        currentPropositionFeatureSet.append(1)
                    else:
                        currentPropositionFeatureSet.append(0)
                    
                    # Check Named Entities (NEs) repetitions inside current proposition
                    if len(set(currentPropositionNamedEntityList)) < len(currentPropositionNamedEntityList):
                        currentPropositionFeatureSet.append(1)
                    else:
                        currentPropositionFeatureSet.append(0)
                    
                elif self.featureSetConfiguration >= 2:
                    # exact match between word lemmas in adjacent propositions (window size of 1)
                    
                    # previous sentence info
                    previousSentence= []
                    previousPropositionAdjectiveList= []
                    previousPropositionNounList= []
                    previousPropositionVerbList= []
                    previousPropositionNamedEntityList= []
                    previousPropositionMainWords= []
                    
                    # previous Proposition
                    if not (int(currentSentence["sentenceId"]) == 0):
                        # previous sentence from database
                        previousSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": int(learningInstance[1]) - 1}]})
                        
                        previousPropositionAdjectiveList= [t["lemma"] for t in previousSentence["tokens"] if (t["tags"][0] == 'J')]
                        
                        previousPropositionNounList= [t["lemma"] for t in previousSentence["tokens"] if (t["tags"] in posTagNouns)]
                        
                        previousPropositionVerbList= [t["lemma"] for t in previousSentence["tokens"] if (t["tags"][0] == 'V')]
                        
                        previousPropositionNamedEntityList= [t["lemma"] for t in previousSentence["tokens"] if (t["tags"] in posTagProperNoun)]
                        
                        previousPropositionMainTokens= [tok for tok in previousSentence["tokens"] if (tok["tags"] in posTagMainWordsList)]
                        
                        previousPropositionMainWords= list(set([tok["lemma"] for tok in previousPropositionMainTokens if not (tok["tags"] in posTagProperNoun)]))
                        
                    
                    
                    # next sentence info
                    # next sentence from database
                    nextSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": int(learningInstance[1]) + 1}]})
                    
                    nextPropositionAdjectiveList= []
                    nextPropositionNounList= []
                    nextPropositionVerbList= []
                    nextPropositionNamedEntityList= []
                    nextPropositionMainWords= []
                    
                    if nextSentence is not None:
                        
                        nextPropositionAdjectiveList= [t["lemma"] for t in nextSentence["tokens"] if (t["tags"][0] == 'J')]
                        
                        nextPropositionNounList= [t["lemma"] for t in nextSentence["tokens"] if (t["tags"] in posTagNouns)]
                        
                        nextPropositionVerbList= [t["lemma"] for t in nextSentence["tokens"] if (t["tags"][0] == 'V')]
                        
                        nextPropositionNamedEntityList= [t["lemma"] for t in nextSentence["tokens"] if (t["tags"] in posTagProperNoun)]
                        
                        nextPropositionMainTokens= [tok for tok in nextSentence["tokens"] if (tok["tags"] in posTagMainWordsList)]
                        
                        nextPropositionMainWords= list(set([tok["lemma"] for tok in nextPropositionMainTokens if not (tok["tags"] in posTagProperNoun)]))
                        
                    
                    
                    adjectiveRepetitionCurrentAndPreviousProposition= False
                    nounRepetitionCurrentAndPreviousProposition= False
                    verbRepetitionCurrentAndPreviousProposition= False
                    namedEntityRepetitionCurrentAndPreviousProposition= False
                    
                    adjectiveRepetitionCurrentAndNextProposition= False
                    nounRepetitionCurrentAndNextProposition= False
                    verbRepetitionCurrentAndNextProposition= False
                    namedEntityRepetitionCurrentAndNextProposition= False
                    
                    if self.featureSetConfiguration == 2:
                        # Check if there is Adjective (A) repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionAdjectiveList)):
                            for z in xrange(len(previousPropositionAdjectiveList)):
                                
                                if currentPropositionAdjectiveList[y] == previousPropositionAdjectiveList[z]:
                                    adjectiveRepetitionCurrentAndPreviousProposition= True
                                    break
                                
                            
                            for z in xrange(len(nextPropositionAdjectiveList)):
                                
                                if currentPropositionAdjectiveList[y] == nextPropositionAdjectiveList[z]:
                                    adjectiveRepetitionCurrentAndNextProposition= True
                                    break
                                
                        
                        if adjectiveRepetitionCurrentAndPreviousProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                            
                        
                        if adjectiveRepetitionCurrentAndNextProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        
                        # Check if there is Noun (N) repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionNounList)):
                            
                            for z in xrange(len(previousPropositionNounList)):
                                
                                if currentPropositionNounList[y] == previousPropositionNounList[z]:
                                    nounRepetitionCurrentAndPreviousProposition= True
                                    break
                                
                            
                            for z in xrange(len(nextPropositionNounList)):
                                
                                if currentPropositionNounList[y] == nextPropositionNounList[z]:
                                    nounRepetitionCurrentAndNextProposition= True
                                    break
                                
                        
                        
                        if nounRepetitionCurrentAndPreviousProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        if nounRepetitionCurrentAndNextProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        # Check if there is Verb (V) repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionVerbList)):
                            
                            for z in xrange(len(previousPropositionVerbList)):
                                
                                if currentPropositionVerbList[y] == previousPropositionVerbList[z]:
                                    verbRepetitionCurrentAndPreviousProposition= True
                                    break
                                
                            
                            for z in xrange(len(nextPropositionVerbList)):
                                
                                if currentPropositionVerbList[y] == nextPropositionVerbList[z]:
                                    verbRepetitionCurrentAndNextProposition= True
                                    break
                                
                        
                        if verbRepetitionCurrentAndPreviousProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        if verbRepetitionCurrentAndNextProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        # Check if there is Name Entity repetitions between current proposition and (previous proposition and next proposition)
                        
                        
                        for y in xrange(len(currentPropositionNamedEntityList)):
                            
                            for z in xrange(len(previousPropositionNamedEntityList)):
                                
                                if currentPropositionNamedEntityList[y] == previousPropositionNamedEntityList[z]:
                                    namedEntityRepetitionCurrentAndPreviousProposition= True
                                    break
                                
                            
                            for z in xrange(len(nextPropositionNamedEntityList)):
                                
                                if currentPropositionNamedEntityList[y] == nextPropositionNamedEntityList[z]:
                                    namedEntityRepetitionCurrentAndNextProposition= True
                                    break
                                
                        
                        if namedEntityRepetitionCurrentAndPreviousProposition:
                            currentPropositionFeatureSet.append(1)
                            
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        if namedEntityRepetitionCurrentAndNextProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        
                    
                    elif self.featureSetConfiguration == 3:
                        # match between word lemmas in adjacent propositions (window size of 1) is made using a word2vec model
                        
                        
                        # Check if there is Adjective (A) repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionAdjectiveList)):
                            
                            for z in xrange(len(previousPropositionAdjectiveList)):
                                if (currentPropositionAdjectiveList[y] in self.wordEmbeddingsModel) and (previousPropositionAdjectiveList[z] in self.wordEmbeddingsModel):
                                    
                                    #TODO Is this threshold the one that yields the best results?
                                    if ((self.wordEmbeddingsModel).distances(currentPropositionAdjectiveList[y], [previousPropositionAdjectiveList[z]]))[0] <= 1:
                                        adjectiveRepetitionCurrentAndPreviousProposition= True
                                        break
                                
                            
                            for z in xrange(len(nextPropositionAdjectiveList)):
                                
                                #TODO Is this threshold the one that yields the best results?
                                if (currentPropositionAdjectiveList[y] in self.wordEmbeddingsModel) and (nextPropositionAdjectiveList[z] in self.wordEmbeddingsModel):
                                    if ((self.wordEmbeddingsModel).distances(currentPropositionAdjectiveList[y], [nextPropositionAdjectiveList[z]]))[0] <= 1:
                                        adjectiveRepetitionCurrentAndNextProposition= True
                                        break
                                
                        
                        if adjectiveRepetitionCurrentAndPreviousProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                            
                        
                        if adjectiveRepetitionCurrentAndNextProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        
                        # Check if there is Noun (N) repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionNounList)):
                            
                            for z in xrange(len(previousPropositionNounList)):
                                
                                #TODO Is this threshold the one that yields the best results?
                                if (currentPropositionNounList[y] in self.wordEmbeddingsModel) and (previousPropositionNounList[z] in self.wordEmbeddingsModel):
                                    if ((self.wordEmbeddingsModel).distances(currentPropositionNounList[y], [previousPropositionNounList[z]]))[0] <= 1:
                                        nounRepetitionCurrentAndPreviousProposition= True
                                        break
                                
                            
                            for z in xrange(len(nextPropositionNounList)):
                                
                                #TODO Is this threshold the one that yields the best results?
                                if (currentPropositionNounList[y] in self.wordEmbeddingsModel) and (nextPropositionNounList[z] in self.wordEmbeddingsModel):
                                    if ((self.wordEmbeddingsModel).distances(currentPropositionNounList[y], [nextPropositionNounList[z]]))[0] <= 1:
                                        nounRepetitionCurrentAndNextProposition= True
                                        break
                                
                        
                        
                        if nounRepetitionCurrentAndPreviousProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        if nounRepetitionCurrentAndNextProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        # Check if there is Verb (V) repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionVerbList)):
                            
                            for z in xrange(len(previousPropositionVerbList)):
                                
                                #TODO Is this threshold the one that yields the best results?
                                if (currentPropositionVerbList[y] in self.wordEmbeddingsModel) and (previousPropositionVerbList[z] in self.wordEmbeddingsModel):
                                    if ((self.wordEmbeddingsModel).distances(currentPropositionVerbList[y], [previousPropositionVerbList[z]]))[0] <= 1:
                                        verbRepetitionCurrentAndPreviousProposition= True
                                        break
                                
                            
                            for z in xrange(len(nextPropositionVerbList)):
                                
                                #TODO Is this threshold the one that yields the best results?
                                if (currentPropositionVerbList[y] in self.wordEmbeddingsModel) and (nextPropositionVerbList[z] in self.wordEmbeddingsModel):
                                    if ((self.wordEmbeddingsModel).distances(currentPropositionVerbList[y], [nextPropositionVerbList[z]]))[0] <= 1:
                                        verbRepetitionCurrentAndNextProposition= True
                                        break
                                
                        
                        if verbRepetitionCurrentAndPreviousProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        if verbRepetitionCurrentAndNextProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        # Check if there is Name Entity repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionNamedEntityList)):
                            #TODO Instead of exact match, check if substring match yields better results (e.g. John Smith vs Smith)
                            
                            for z in xrange(len(previousPropositionNamedEntityList)):
                                if currentPropositionNamedEntityList[y] == previousPropositionNamedEntityList[z]:
                                    nameEntityRepetitionCurrentAndPreviousProposition= True
                                    break
                                
                            
                            for z in xrange(len(nextPropositionNamedEntityList)):
                                if currentPropositionNamedEntityList[y] == nextPropositionNamedEntityList[z]:
                                    nameEntityRepetitionCurrentAndNextProposition= True
                                    break
                                
                        
                        if nameEntityRepetitionCurrentAndPreviousProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        if nameEntityRepetitionCurrentAndNextProposition:
                            currentPropositionFeatureSet.append(1)
                        else:
                            currentPropositionFeatureSet.append(0)
                        
                        
                        
                    elif self.featureSetConfiguration == 4:
                        # calculates the minimum distance between each pair of words in the corresponding PoS tags between current proposition and previous and between 
                        # current proposition and next proposition
                        #TODO: This feature does not seem very interesting -> in the limit it just indicates that the propositions have at least one word in common or not
                        
                        distancePreviousProposition= 100
                        distanceNextProposition= 100
                        
                        # Check if there is Adjective (A) repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionAdjectiveList)):
                            for z in xrange(len(previousPropositionAdjectiveList)):
                                if (currentPropositionAdjectiveList[y] in self.wordEmbeddingsModel) and (previousPropositionAdjectiveList[z] in self.wordEmbeddingsModel):
                                    currentValue= ((self.wordEmbeddingsModel).distances(currentPropositionAdjectiveList[y], [previousPropositionAdjectiveList[z]]))[0]
                                    if currentValue <= distancePreviousProposition:
                                        distancePreviousProposition= currentValue
                                
                            
                            for z in xrange(len(nextPropositionAdjectiveList)):
                                if (currentPropositionAdjectiveList[y] in self.wordEmbeddingsModel) and (nextPropositionAdjectiveList[z] in self.wordEmbeddingsModel):
                                    currentValue= ((self.wordEmbeddingsModel).distances(currentPropositionAdjectiveList[y], [nextPropositionAdjectiveList[z]]))[0]
                                    if currentValue <= distanceNextProposition:
                                        distanceNextProposition= currentValue
                                
                        
                        
                        
                        # Check if there is Noun (N) repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionNounList)):
                            for z in xrange(len(previousPropositionNounList)):
                                if (currentPropositionNounList[y] in self.wordEmbeddingsModel) and (previousPropositionNounList[z] in self.wordEmbeddingsModel):
                                    currentValue= ((self.wordEmbeddingsModel).distances(currentPropositionNounList[y], [previousPropositionNounList[z]]))[0]
                                    if currentValue <= distancePreviousProposition:
                                        distancePreviousProposition= currentValue
                                        #break
                                
                            
                            for z in xrange(len(nextPropositionNounList)):
                                if (currentPropositionNounList[y] in self.wordEmbeddingsModel) and (nextPropositionNounList[z] in self.wordEmbeddingsModel):
                                    currentValue= ((self.wordEmbeddingsModel).distances(currentPropositionNounList[y], [nextPropositionNounList[z]]))[0]
                                    if currentValue <= distanceNextProposition:
                                        distanceNextProposition= currentValue
                                        #break
                                
                        
                        
                        
                        # Check if there is Verb (V) repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionVerbList)):
                            for z in xrange(len(previousPropositionVerbList)):
                                if (currentPropositionVerbList[y] in self.wordEmbeddingsModel) and (previousPropositionVerbList[z] in self.wordEmbeddingsModel):
                                    currentValue= ((self.wordEmbeddingsModel).distances(currentPropositionVerbList[y], [previousPropositionVerbList[z]]))[0]
                                    if currentValue <= distancePreviousProposition:
                                        distancePreviousProposition= currentValue
                                        
                                
                            
                            for z in xrange(len(nextPropositionVerbList)):
                                if (currentPropositionVerbList[y] in self.wordEmbeddingsModel) and (nextPropositionVerbList[z] in self.wordEmbeddingsModel):
                                    currentValue= ((self.wordEmbeddingsModel).distances(currentPropositionVerbList[y], [nextPropositionVerbList[z]]))[0]
                                    if currentValue <= distanceNextProposition:
                                        distanceNextProposition= currentValue
                                        
                                
                        
                        
                        # Check if there is Name Entity repetitions between current proposition and (previous proposition and next proposition)
                        
                        for y in xrange(len(currentPropositionNamedEntityList)):
                            for z in xrange(len(previousPropositionNamedEntityList)):
                                if (currentPropositionNamedEntityList[y] in self.wordEmbeddingsModel) and (previousPropositionNamedEntityList[z] in self.wordEmbeddingsModel):
                                    currentValue= ((self.wordEmbeddingsModel).distances(currentPropositionNamedEntityList[y], [previousPropositionNamedEntityList[z]]))[0]
                                    if currentValue <= distancePreviousProposition:
                                        distancePreviousProposition= currentValue
                                        
                                
                            
                            for z in xrange(len(nextPropositionNamedEntityList)):
                                if (currentPropositionNamedEntityList[y] in self.wordEmbeddingsModel) and (nextPropositionNamedEntityList[z] in self.wordEmbeddingsModel):
                                    currentValue= ((self.wordEmbeddingsModel).distances(currentPropositionNamedEntityList[y], [nextPropositionNamedEntityList[z]]))[0]
                                    if currentValue <= distanceNextProposition:
                                        distanceNextProposition= currentValue
                                
                        
                        currentPropositionFeatureSet.append(distancePreviousProposition)
                        currentPropositionFeatureSet.append(distanceNextProposition)
                        
                    
                    elif self.featureSetConfiguration == 5:
                        
                        # determine words repetitions
                        repeatedtokensBetweenCurrentAndPreviousProposition= list(set(currentPropositionMainWords).intersection(set(previousPropositionMainWords)))
                        
                        # determine words repetitions
                        repeatedtokensBetweenCurrentAndNextProposition= list(set(currentPropositionMainWords).intersection(set(nextPropositionMainWords)))
                        
                        #currentPropositionNEs= [tok.split("_") for tok in currentPropositionNEs]
                        
                        ### Current -> Previous ###
                        # overlap of words between current and previous proposition in relation to the words in the current proposition
                        overlapCurrentPreviousProposition_Current= 0.0
                        synonymsOverlapCurrentPreviousProposition_Current= 0
                        hyperonymsOverlapCurrentPreviousProposition_Current= 0
                        hyponymsOverlapCurrentPreviousProposition_Current= 0
                        antonymsOverlapCurrentPreviousProposition_Current= 0
                        nesOverlapCurrentPreviousProposition_Current= 0
                        
                        
                        # tokens exact match overlap
                        
                        if len(currentPropositionMainWords) > 0:
                            overlapCurrentPreviousProposition_Current= float(len(repeatedtokensBetweenCurrentAndPreviousProposition)) / float(len(currentPropositionMainWords))
                        #overlapCurrentPreviousProposition_Previous= float(len(repeatedtokensBetweenCurrentAndPreviousProposition)) / float(len(previousPropositionMainWords))
                        
                        
                        # Proposition unique main words - main words occurring in proposition that do not occur in the other proposition
                        currentPropositionMainWords= [tokLemma for tokLemma in currentPropositionMainWords if not (tokLemma in repeatedtokensBetweenCurrentAndPreviousProposition)]
                        previousPropositionMainWords= [tokLemma for tokLemma in previousPropositionMainWords if not (tokLemma in repeatedtokensBetweenCurrentAndPreviousProposition)]
                        
                        
                        # synonyms overlap
                        numSynonymsInCurrentPropositionFromPreviousProposition = 0
                        
                        for currentPropositionLemma in currentPropositionMainWords:
                            
                            for previousPropositionLemma in previousPropositionMainWords:
                                
                                if ((self.wordnet).isSynonyn(currentPropositionLemma, previousPropositionLemma)):
                                    
                                    numSynonymsInCurrentPropositionFromPreviousProposition += 1
                                    break
                        
                        
                        
                        if len(currentPropositionMainWords) > 0:
                            synonymsOverlapCurrentPreviousProposition_Current= float(numSynonymsInCurrentPropositionFromPreviousProposition) / len(currentPropositionMainWords)
                        
                        
                        
                        # hyperonyms overlap
                        hyperonymsOverlapInCurrentProposition= 0
                        
                        
                        for currentPropositionLemma in currentPropositionMainWords:
                            
                            for previousPropositionLemma in previousPropositionMainWords:
                                
                                if ((self.wordnet).isHyperonym(currentPropositionLemma, previousPropositionLemma)) or ((self.wordnet).isHyperonym(previousPropositionLemma, currentPropositionLemma)):
                                    
                                    hyperonymsOverlapInCurrentProposition += 1
                                    break
                        
                        
                        
                        if len(currentPropositionMainWords) > 0:
                            hyperonymsOverlapCurrentPreviousProposition_Current= float(hyperonymsOverlapInCurrentProposition) / len(currentPropositionMainWords)
                        
                        
                        
                        
                        # meronyms overlap
                        hyponymsOverlapInCurrentProposition= 0
                        
                        for currentPropositionLemma in currentPropositionMainWords:
                            
                            for previousPropositionLemma in previousPropositionMainWords:
                                
                                if ((self.wordnet).isHyponym(currentPropositionLemma, previousPropositionLemma)) or ((self.wordnet).isHyponym(previousPropositionLemma, currentPropositionLemma)):
                                    
                                    hyponymsOverlapInCurrentProposition += 1
                                    break
                        
                        
                        
                        if len(currentPropositionMainWords) > 0:
                            hyponymsOverlapCurrentPreviousProposition_Current= float(hyponymsOverlapInCurrentProposition) / len(currentPropositionMainWords)
                        
                        
                        
                        
                        # antonyms overlap
                        antonymsOverlapInCurrentProposition= 0
                        
                        for currentPropositionLemma in currentPropositionMainWords:
                            
                            for previousPropositionLemma in previousPropositionMainWords:
                                
                                if ((self.wordnet).isAntonym(currentPropositionLemma, previousPropositionLemma)) or ((self.wordnet).isAntonym(previousPropositionLemma, currentPropositionLemma)):
                                    
                                    antonymsOverlapInCurrentProposition += 1
                                    break
                        
                        
                        
                        if len(currentPropositionMainWords) > 0:
                            antonymsOverlapCurrentPreviousProposition_Current= float(antonymsOverlapInCurrentProposition) / len(currentPropositionMainWords)
                        
                        
                        # Named Entities (NEs) overlap
                        nesOverlapInCurrentProposition= 0
                        
                        for currentPropositionNamedEntity in currentPropositionNamedEntityList:
                            
                            for previousPropositionNamedEntity in previousPropositionNamedEntityList:
                                
                                if currentPropositionNamedEntity == previousPropositionNamedEntity:
                                    nesOverlapInCurrentProposition += 1
                                    break
                                
                                
                            
                        
                        if len(currentPropositionNamedEntityList) > 0:
                            nesOverlapCurrentPreviousProposition_Current= float(nesOverlapInCurrentProposition) / len(currentPropositionNamedEntityList)
                        
                        
                        # add overlap scores to feature set
                        currentPropositionFeatureSet.append(overlapCurrentPreviousProposition_Current)
                        currentPropositionFeatureSet.append(synonymsOverlapCurrentPreviousProposition_Current)
                        currentPropositionFeatureSet.append(hyperonymsOverlapCurrentPreviousProposition_Current)
                        currentPropositionFeatureSet.append(hyponymsOverlapCurrentPreviousProposition_Current)
                        currentPropositionFeatureSet.append(antonymsOverlapCurrentPreviousProposition_Current)
                        currentPropositionFeatureSet.append(nesOverlapCurrentPreviousProposition_Current)
                        
                        
                        ### Current -> Next ###
                        overlapCurrentNextProposition_Current= 0.0
                        synonymsOverlapCurrentNextProposition_Current= 0
                        hyperonymsOverlapCurrentNextProposition_Current= 0
                        hyponymsOverlapCurrentNextProposition_Current= 0
                        antonymsOverlapCurrentNextProposition_Current= 0
                        nesOverlapCurrentNextProposition_Current= 0
                        
                        
                        
                        # tokens exact match overlap
                        if len(currentPropositionMainWords) > 0:
                            overlapCurrentNextProposition_Current= float(len(repeatedtokensBetweenCurrentAndNextProposition)) / float(len(currentPropositionMainWords))
                        #overlapCurrentPreviousProposition_Previous= float(len(repeatedtokensBetweenCurrentAndPreviousProposition)) / float(len(previousPropositionMainWords))
                        
                        
                        # Proposition unique main words - main words occurring in proposition that do not occur in the other proposition
                        currentPropositionMainWords= [tokLemma for tokLemma in currentPropositionMainWords if not (tokLemma in repeatedtokensBetweenCurrentAndNextProposition)]
                        nextPropositionMainWords= [tokLemma for tokLemma in nextPropositionMainWords if not (tokLemma in repeatedtokensBetweenCurrentAndNextProposition)]
                        
                        # synonyms overlap
                        numSynonymsInCurrentPropositionFromNextProposition = 0
                        
                        for currentPropositionLemma in currentPropositionMainWords:
                            
                            for nextPropositionLemma in nextPropositionMainWords:
                                
                                if ((self.wordnet).isSynonyn(currentPropositionLemma, nextPropositionLemma)):
                                    
                                    numSynonymsInCurrentPropositionFromNextProposition += 1
                                    break
                        
                        
                        
                        if len(currentPropositionMainWords) > 0:
                            synonymsOverlapCurrentNextProposition_Current= float(numSynonymsInCurrentPropositionFromNextProposition) / len(currentPropositionMainWords)
                        
                        
                        
                        # hyperonyms overlap
                        hyperonymsOverlapInCurrentProposition= 0
                        
                        
                        for currentPropositionLemma in currentPropositionMainWords:
                            
                            for nextPropositionLemma in nextPropositionMainWords:
                                
                                if ((self.wordnet).isHyperonym(currentPropositionLemma, nextPropositionLemma)) or ((self.wordnet).isHyperonym(nextPropositionLemma, currentPropositionLemma)):
                                    
                                    hyperonymsOverlapInCurrentProposition += 1
                                    break
                        
                        
                        
                        if len(currentPropositionMainWords) > 0:
                            hyperonymsOverlapCurrentNextProposition_Current= float(hyperonymsOverlapInCurrentProposition) / len(currentPropositionMainWords)
                        
                        
                        
                        
                        # meronyms overlap
                        
                        hyponymsOverlapInCurrentProposition= 0
                        
                        for currentPropositionLemma in currentPropositionMainWords:
                            
                            for nextPropositionLemma in nextPropositionMainWords:
                                
                                if ((self.wordnet).isHyponym(currentPropositionLemma, nextPropositionLemma)) or ((self.wordnet).isHyponym(nextPropositionLemma, currentPropositionLemma)):
                                    
                                    hyponymsOverlapInCurrentProposition += 1
                                    break
                        
                        
                        
                        if len(currentPropositionMainWords) > 0:
                            hyponymsOverlapCurrentNextProposition_Current= float(hyponymsOverlapInCurrentProposition) / len(currentPropositionMainWords)
                        
                        
                        
                        
                        # antonyms overlap
                        
                        antonymsOverlapInCurrentProposition= 0
                        
                        for currentPropositionLemma in currentPropositionMainWords:
                            
                            for nextPropositionLemma in nextPropositionMainWords:
                                
                                if ((self.wordnet).isAntonym(currentPropositionLemma, nextPropositionLemma)) or ((self.wordnet).isAntonym(nextPropositionLemma, currentPropositionLemma)):
                                    
                                    antonymsOverlapInCurrentProposition += 1
                                    break
                        
                        
                        
                        if len(currentPropositionMainWords) > 0:
                            antonymsOverlapCurrentNextProposition_Current= float(antonymsOverlapInCurrentProposition) / len(currentPropositionMainWords)
                        
                        
                        # Named Entities (NEs) overlap
                        nesOverlapInCurrentProposition= 0
                        
                        
                        for currentPropostionNamedEntity in currentPropositionNamedEntityList:
                            
                            for nextPropositionNamedEntity in nextPropositionNamedEntityList:
                                
                                
                                if currentPropostionNamedEntity == nextPropositionNamedEntity:
                                    nesOverlapInCurrentProposition += 1
                                    break
                                
                                
                            
                        
                        if len(currentPropositionNamedEntityList) > 0:
                            nesOverlapCurrentNextProposition_Current= float(nesOverlapInCurrentProposition) / len(currentPropositionNamedEntityList)
                        
                        
                        
                        # add overlap scores to feature set
                        currentPropositionFeatureSet.append(overlapCurrentNextProposition_Current)
                        currentPropositionFeatureSet.append(synonymsOverlapCurrentNextProposition_Current)
                        currentPropositionFeatureSet.append(hyperonymsOverlapCurrentNextProposition_Current)
                        currentPropositionFeatureSet.append(hyponymsOverlapCurrentNextProposition_Current)
                        currentPropositionFeatureSet.append(antonymsOverlapCurrentNextProposition_Current)
                        currentPropositionFeatureSet.append(nesOverlapCurrentNextProposition_Current)
                        
                    
                    
                
                
                self.featureArray.append(np.asarray(currentPropositionFeatureSet))

            # close connection
            mongoClient.close()
        
        return np.asarray(self.featureArray)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        if self.featureSetConfiguration == 0:
            return ["None"]
        elif self.featureSetConfiguration == 1:
            return ['CurrentPropositionAdjRepetition', 'CurrentPropositionNounRepetition', 'CurrentPropositionVerbRepetition', 'CurrentPropositionNameEntityRepetition']
        elif (self.featureSetConfiguration == 2) or (self.featureSetConfiguration == 3):
            featuresNamesList= []
            
            featuresNamesList= featuresNamesList + ['CurrentPreviousPropositionAdjRepetition', 'CurrentFollowingPropositionAdjRepetition']
            featuresNamesList= featuresNamesList + ['CurrentPreviousPropositionNounRepetition', 'CurrentFollowingPropositionNounRepetition']
            featuresNamesList= featuresNamesList + ['CurrentPreviousPropositionVerbRepetition', 'CurrentFollowingPropositionVerbRepetition']
            featuresNamesList= featuresNamesList + ['CurrentPreviousPropositionNameEntityRepetition', 'CurrentFollowingPropositionNameEntityRepetition']
            
            return featuresNamesList
        elif self.featureSetConfiguration == 4:
            return ['distancePreviousProposition', 'distanceNextProposition']
        elif self.featureSetConfiguration == 5:
            return ['exactMatchOverlap_Previous', 'synonymsOverlap_Previous', 'hyperonymOverlap_Previous', 'meronymOverlap_Previous', 'antonymOverlap_Previous', 'namedEntitiesOverlap_Previous', 'exactMatchOverlap_Next', 'synonymsOverlap_Next', 'hyperonymOverlap_Next', 'meronymOverlap_Next', 'antonymOverlap_Next', 'namedEntitiesOverlap_Next']

    def get_content(self):
        return np.asarray(self.featureArray)
