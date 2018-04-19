#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WordCouplesTransformer: Custom Transformer to create the Word Couples features.
Receives as input a matrix with dimensions 1:X where each element is a sentence/proposition
(being X the number of propositions in the fold). Outputs a matrix with dimension 
X:Y, where Y is the number of Word Couples existing in the dataset. This feature set is a customized 
Bag of Words, without the restriction of adjacent pairs of words and constrained (or not) to pairs of 
one or more argumentative keywords. Each element is an integer 
indicating the number of times the corresponding word couple appears in the corresponding proposition
"""

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from pymongo import MongoClient
import string
from utils.Parameters import Parameters
parameters= Parameters()

class WordCouplesTransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, numberOfKeywords= 2, cleanCorpus= True):
        
        # vocabulary of all pairs of tokens (word couples) existing in the dataset
        # where the keys are the word couples and the values are consecutive integers 
        # representing the feature id in the Term-Document Matrix for this word couple
        self.vocabulary_ = {}
        
        # Term-Document Matrix -> where 'Documents' are propositions in this case and 'Terms' 
        # are the different word couples existing in the dataset
        # the value of each element is an integer indicating the number of times the corresponding 
        # word couple appears in the corresponding proposition
        self.termDocumentMatrix= []
        
        # Boolean variable indication whether we should clean the corpus (True) or not (False)
        self.cleanCorpus= cleanCorpus
        
        # variable indicating if we should update vocabulary in the method "transform" or if we should maintain current vocabulary
        self.fixedVocabulary_= False
        
        
        # integer variable indicating the number of keywordsmodel that WordCouples should have
        # If 0, then we want all possible WordCouples without any restriction
        # If 1, then we want WordCouples where at least on of the elements is keyword
        # If 2, then we want WordCouples where both elements are keywordsmodel.
        if (numberOfKeywords > 3) or (numberOfKeywords < 0):
            raise Exception("Number of keywords invalid! NumberOfKeywords= " + str(numberOfKeywords) + " but should be in the range [0, 3]" )
        else:
            self.numberOfKeywords= numberOfKeywords
        
        if self.numberOfKeywords > 0:
            
            self.keywords= []
            
            premisseKeywordsFile= open(parameters.paths["keywords_en"] + '/' + parameters.filenames["premiseKeywords"],'r')
            
            #premisseKeywordsList= []
            
            for word in premisseKeywordsFile:
                if len(word) > 1:
                    (self.keywords).append(word.rstrip("\n").decode("utf-8"))
                
            conclusionKeywordsFile= open(parameters.paths["keywords_en"] + '/' + parameters.filenames["conclusionKeywords"],'r')
            
            #conclusionKeywordsList= []
            
            for word in conclusionKeywordsFile:
                if len(word) > 1:
                    (self.keywords).append(word.rstrip("\n").decode("utf-8"))
                
            
            
        else:
            self.keywords= None

    def transform(self, X, **transform_params):
        
        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        
        aaecCorpus = mongoClient.AAECCorpus
        
        # Sentence's table
        sentenceCollection= aaecCorpus.sentence
        
        if (self.numberOfKeywords >= 0) and (self.numberOfKeywords <= 2):
            # create vocabulary of pairs of tokens (word couples)
            if not self.fixedVocabulary_ :
                
                for sentenceTuple in X:
                    # current learning instance info from database
                    currentSentence= sentenceCollection.find_one({"$and":[{"articleId": sentenceTuple[0]}, {"sentenceId": sentenceTuple[1]}]})
                    self.updateVocabulary(currentSentence)
                    
                # just updates the vocabulary during training
                # for unseen data it should use only the vocabulary generated during training and ignore new words
                # It is assumed that at this point the vocabulary already has the elements from training data
                
                
                self.fixedVocabulary_= True
            
            
            
            # create Term-Document Matrix
            self.termDocumentMatrix= self.getTermDocumentMatrix(X)
            
            
            # one hot encoding:
            newTermDocumentMatrix= []
            for prop in self.termDocumentMatrix:
                newPropCounts= []
                for wordCoupleCount in prop:
                    if wordCoupleCount > 0:
                        newPropCounts.append(1)
                    else:
                        newPropCounts.append(0)
                newTermDocumentMatrix.append(newPropCounts)
            
            
            self.termDocumentMatrix= np.asarray(newTermDocumentMatrix)
            
        else:
            self.termDocumentMatrix= [[0] for j in range(len(X))]
        
        # close connection
        mongoClient.close()
        
        return self.termDocumentMatrix

    def updateVocabulary(self, proposition):
        
        # obtain all possible combinations of WordCouples from "proposition"
        wordCouplesList= self.getAllCombinationsOfTokens(proposition)
        
        # for each WordCouple add it to the vocabulary if it has some desired characteristics (as defined in the parameters)
        for wordCouple in wordCouplesList:
            
            if wordCouple not in self.vocabulary_:
                
                
                if self.numberOfKeywords == 0:
                    self.vocabulary_[wordCouple] = len(self.vocabulary_)
                else:
                    
                    keywordInFirstWordCouple= False
                    keywordInSecondWordCouple= False
                    
                    if (self.keywords is not None):
                        
                        words= wordCouple.split(" ")
                        
                        if words[0] in self.keywords:
                            keywordInFirstWordCouple = True
                        
                        if words[1] in self.keywords:
                            keywordInSecondWordCouple = True
                        
                    
                    
                    
                    
                    if self.numberOfKeywords == 1:
                        # at least one of the words belonging to the word couple is keyword
                        if keywordInFirstWordCouple or keywordInSecondWordCouple:
                            self.vocabulary_[wordCouple] = len(self.vocabulary_)
                    else:
                        # both words belonging to the word couple are keywordsmodel
                        if keywordInFirstWordCouple and keywordInSecondWordCouple:
                            self.vocabulary_[wordCouple] = len(self.vocabulary_)
    # construct Term Document Matrix -> corresponds to word couples counts existing in each proposition
    # Input: vocabulary (corresponds to the features) and set of proposition
    # return matrix (no. proposition x no. word couples in all the dataset) where each element 
    # corresponds to the number of times a word couples appears in a specific proposition
    def getTermDocumentMatrix(self, X):
        
        if len(self.vocabulary_) == 0:
            # Due to the restrictions applied to the possible extracted word couples, it can happen (in case that the restrictions 
            # are too restrictive) that no word couples were found. As a consequence the feature set is empty, which raises an 
            # exception because it cannot happen in a sklearn transformer 
            m= [[0] for j in range(len(X))]
            
        else:
            
            # initialize Term-Document matrix with zeros
            # matrix dimensions are: [no. of proposition, no. of features] where,
            # no. of features = number of different word couples
            m= [[0 for i in range(len(self.vocabulary_))] for j in range(len(X))]
            
            # update matrix with counts of word couples existing in each proposition
            
            currentIndex= 0
            
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            
            aaecCorpus = mongoClient.AAECCorpus
            
            # Sentence's table
            sentenceCollection= aaecCorpus.sentence
            
            for sentenceTuple in X:
                
                # current learning instance info from database
                currentSentence= sentenceCollection.find_one({"$and":[{"articleId": sentenceTuple[0]}, {"sentenceId": sentenceTuple[1]}]})
                
                wordCouplesList= self.getAllCombinationsOfTokens(currentSentence)
                
                for wordPair in wordCouplesList:
                    
                    
                    if wordPair in self.vocabulary_:
                        # obtain column index corresponding to this word pair (feature index)
                        column= self.vocabulary_[wordPair]
                        m[currentIndex][column] += 1
                
                currentIndex += 1
            
            # close database connection
            mongoClient.close()
            
        
        return np.asarray(m)
    # given a proposition returns all combinations of tokens pairs
    # this pairs are constructed in the following way: one token is the "pivot" and the pairs are 
    # constructed combining this pivot with all the tokens that exist after the pivot in the proposition
    # all the tokens in the proposition are pivots, one at a time (in order to obtain all 
    # pairs of tokens existing on the proposition) 
    # repeated tokens pairs are allowed (because this information is important when constructing 
    # the Term-Document Matrix)
    def getAllCombinationsOfTokens(self, proposition):
        
        tokensList= proposition["tokens"]
        
        
        if self.cleanCorpus:
            tokensList= self.removeStopWords(tokensList)
        
        wordCouples= []
        
        
        for i in range(len(tokensList)):
            
            
            # current word (also called as pivot) -> corresponds to the first word/token
            # in any word couple
            currentWord= tokensList[i]["lemma"]
            
            # j belongs to [i+1, proposition length]
            # token/word couples are all combinations of the current token (pivot) and all 
            # the tokens that exist after this token on the sentence
            for j in range(i + 1, len(tokensList)):
                currentWordCouple= currentWord + " " + tokensList[j]["lemma"]
                
                wordCouples.append(currentWordCouple)
        
        
        
        return wordCouples
    
    def removeStopWords(self, tokensList):
        
        punctuationMarks= string.punctuation
        punctuationMarksList= [punctuationMarks[i] for i in range(len(punctuationMarks))]
        
        posTagProperNoun= ['NNP', 'NNPS']
        
        stopWordsFile= open(parameters.paths["keywords_en"] + '/' + parameters.filenames["stopWords"],'r')
        
        stopWordsList= []
        
        for word in stopWordsFile:
            stopWordsList.append(word.rstrip("\n").decode("utf-8"))
        
        # list of tokens after removing: punctuation marks, stop words, proper nouns and numbers
        tokensList = [t for t in tokensList if not ( (t["lemma"] in punctuationMarksList) or (t["lemma"] in stopWordsList ) or ( t["tags"][0] in posTagProperNoun ) or (t["tags"] == 'CD' ) )]
        
        return tokensList

    def fit(self, X, y=None, **fit_params):
        return self

    def vocabulary_(self):
        return self.vocabulary_ #.keys()
    
    def get_content(self):
        return self.termDocumentMatrix
    
    def get_feature_names(self):
        return self.vocabulary_.keys()
    # override method from BaseEstimator class
    # the parameter self.keywordsmodel should be updated according to updates in the parameter "self.numberOfKeywords"
    # Besides self.numberOfKeywords being updated correctly inheriting this method from BaseEstimator, we have to override this 
    # method to update self.keywordsmodel 
    def set_params(self, **params):
        
        
        numberOfKeywordsBeforeUpdate= self.numberOfKeywords
        
        # call inherited method from BaseEstimator class to update some of the parameters
        super(WordCouplesTransformer, self).set_params(**params)
        
        
        
        # integer variable indicating the number of keywordsmodel that WordCouples should have
        # If 0, then we want all possible WordCouples without any restriction
        # If 1, then we want WordCouples where at least on of the elements is keyword
        # If 2, then we want WordCouples where both elements are keywordsmodel.
        if (self.numberOfKeywords > 3) or (self.numberOfKeywords < 0):
            raise Exception("Number of keywords invalid! NumberOfKeywords= " + str(self.numberOfKeywords) + " but should be in the range [0, 3]" )
        
        # update set of keywordsmodel (self.keywordsmodel)
        if (self.numberOfKeywords > 0) and (numberOfKeywordsBeforeUpdate == 0):
            
            self.keywords= []
            
            premisseKeywordsFile= open(parameters.paths["keywords_en"] + '/' + parameters.filenames["premiseKeywords"],'r')
            
            #premisseKeywordsList= []
            
            for word in premisseKeywordsFile:
                if len(word) > 1:
                    (self.keywords).append(word.rstrip("\n").decode("utf-8"))
                
            conclusionKeywordsFile= open(parameters.paths["keywords_en"] + '/' + parameters.filenames["conclusionKeywords"],'r')
            
            #conclusionKeywordsList= []
            
            for word in conclusionKeywordsFile:
                if len(word) > 1:
                    (self.keywords).append(word.rstrip("\n").decode("utf-8"))
                
            
            
        else:
            self.keywords= None
