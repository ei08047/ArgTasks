"""
SyntacticalTransformer: Custom Transformer with the purpose to create syntactic level features.
It receives as input a matrix with dimensions 1:X where each element is a sentence/proposition
(being X the number of propositions in the dataset). It outputs a matrix with dimension 
X:Y, where:
Y= A + B + C,
A = if  modalAuxiliaryFeature is True, then # modal auxiliaries presented in X, otherwise 0,
B = if  verbsFeature is True, then # verbs presented in X, otherwise 0,
C = if  adverbsFeature is True, then # adverbs presented in X, otherwise 0
The feature set Y corresponds to the disjoint union of the individual feature sets A, B and C, with the features of A followed 
by the features of B and the features B followed by the features of C (features are not mixed).
Each element is an integer indicating the number of times the corresponding word couple appears in the corresponding proposition
"""

import string
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from pymongo import MongoClient

from utils.Parameters import Parameters
parameters= Parameters()

VERB= 'V'
ADVERB= 'R'


class SyntacticalTransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, modalAuxiliaryFeature= True, verbsFeature= False, adverbsFeature= True, cleanCorpus= True):
        # vocabulary of all tokens belonging to one the the categories (verb, adverb or modal auxiliary) existing in the dataset
        # the keys are tokens and the values are sequential integers (this values will be used as feature id's to generate the 
        # Term-Document Matrix) 
        self.vocabulary_ = {}
        
        # Term-Document Matrix -> where 'Documents' are propositions in this case and 'Terms' 
        # are the different tokens existing in the dataset
        # the value of each element is an integer indicating the number of times the corresponding 
        # token appears in the corresponding proposition
        self.termDocumentMatrix= []
        
        # Boolean variable indication whether we should clean the corpus (True) or not (False)
        self.cleanCorpus= cleanCorpus
        
        # variable indicating if we should update vocabulary in the method "transform" or if we should maintain current vocabulary
        self.fixedVocabulary_= False
        
        # Boolean indicating if we are interested in verbs as features
        self.verbsFeature= verbsFeature
        
        # Set of features, where each feature corresponds to a verb
        self.verbsFeatureSet_= {}
        
        # Boolean indicating if we are interested in adverbs as features
        self.adverbsFeature= adverbsFeature
        
        # Set of features, where each feature corresponds to an adverb
        self.adverbsFeatureSet_= {}
        
        # Boolean indicating if we are interested in modal auxiliary as features
        self.modalAuxiliaryFeature= modalAuxiliaryFeature
        
        # Set of features, where each feature corresponds to a modal auxiliary
        self.modalAuxiliaryFeatureSet_ = {}
        
        # List of modal auxiliary keywordsmodel -> this list corresponds to the set of words that are considered to be modal auxiliary in the Portuguese language
        self.modalAuxiliaryList= []
        
        if self.modalAuxiliaryFeature:
            
            modalAuxiliariesFile= open(parameters.paths["keywords_en"] + '/' + parameters.filenames["modalAuxiliary"],'r')
            
            
            for word in modalAuxiliariesFile:
                (self.modalAuxiliaryList).append(word.rstrip("\n").decode("utf-8"))

    def transform(self, X, **transform_params):
        
        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        
        aaecCorpus = mongoClient.AAECCorpus
        
        # Sentence's table
        sentenceCollection= aaecCorpus.sentence
        
        # create vocabulary of tokens, where each token should be a modal auxiliar, verb or adverb as specified by the set of 
        # booleans created in the initializer
        if not self.fixedVocabulary_ :
            
            for learningInstance in X:
                # current learning instance info from database
                currentSentence= sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": learningInstance[1]}]})
                
                # update vocabulary for modal auxiliaries, verbs and adverbs
                self.updateSetOfVocabularies(currentSentence["tokens"])
                
                
            
            # update main vocabulary -> disjoint union of each of the individual vocabularies
            self.updateVocabulary()
            
            # just updates the vocabulary during training
            # for unseen data it should use only the vocabulary generated during training and ignore new words
            # It is assumed that at this point the vocabulary already has the elements from training data
            self.fixedVocabulary_= True
        
        # close connection
        mongoClient.close()
        
        # create Term-Document Matrix
        self.termDocumentMatrix= self.getTermDocumentMatrix(X)
        
        return self.termDocumentMatrix

    # construct Term Document Matrix -> corresponds to tokens counts existing in each proposition
    # Input: vocabulary (corresponds to the features) and set of proposition
    # return matrix (no. proposition x no. tokens in all the dataset) where each element 
    # corresponds to the number of times a specific token appears in a specific proposition
    def getTermDocumentMatrix(self, X):
        
        if (not self.modalAuxiliaryFeature) and (not self.verbsFeature) and (not self.adverbsFeature):
            m= [[0] for j in range(len(X))]
        else:
            
            # initialize Term-Document matrix with zeros
            # matrix dimensions are: [no. of proposition, no. of features] where,
            # no. of features = number of different tokens (that are modal auxiliary, verbs or adverbs)
            m= [[0 for i in range(len(self.vocabulary_))] for j in range(len(X))]
            
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            
            aaecCorpus = mongoClient.AAECCorpus
            
            # Sentence's table
            sentenceCollection= aaecCorpus.sentence
            # update matrix with counts of tokens existing in each proposition
            currentIndex= 0
            
            for learningInstance in X:
                
                # current learning instance info from database
                currentSentence= sentenceCollection.find_one({"$and":[{"articleId": int(learningInstance[0])}, {"sentenceId": int(learningInstance[1])}]})
                
                tokensList= self.getAllCombinationsOfTokens(currentSentence["tokens"])
                
                for token in tokensList:
                    
                    if token in self.vocabulary_:
                        # obtain column index corresponding to this token (feature index)
                        column= self.vocabulary_[token]
                        m[currentIndex][column] += 1
                
                currentIndex += 1
            
            # close connection
            mongoClient.close()
        
        return np.asarray(m)

    def updateVocabulary(self):
        
        self.vocabulary_= (self.modalAuxiliaryFeatureSet_).copy()
        
        for k, v in (self.verbsFeatureSet_).items():
            if k not in self.vocabulary_:
                
                self.vocabulary_[k] = len(self.vocabulary_)
        
        
        
        for k, v in (self.adverbsFeatureSet_).items():
            if k not in self.vocabulary_:
                
                self.vocabulary_[k] = len(self.vocabulary_)
        
        #(self.vocabulary_).update(self.adverbsFeatureSet_)
    # given a proposition updates all individual vocabularies with the content of this proposition
    # the individual vocabularies are:
    # 1) self.modalAuxiliaryFeatureSet_ -> vocabulary of modal auxiliary words
    # 2) self.verbsFeature -> vocabulary of verbs
    # 3) self.adverbsFeature -> vocabulary of adverbs
    def updateSetOfVocabularies(self, tokens):
        
        if self.cleanCorpus:
            tokens= self.removeStopWords(tokens)
        
        
        for token in tokens:
            # update individual vocabularies
            # the order is important because if the current token is considered to be in one of the categories it will not update
            # on the others categories. Currently it is in this order because it was considered that the most relevant category 
            # for argumentation is modal auxiliaries, followed by verbs and followed by adverbs. Therefore the feature on the 
            # left side should be the most relevant. However, this was not studied in detail.
            if self.modalAuxiliaryFeature and (token["lemma"] in self.modalAuxiliaryList):
                
                if token["lemma"] not in self.modalAuxiliaryFeatureSet_:
                    
                    self.modalAuxiliaryFeatureSet_[token["lemma"]] = len(self.modalAuxiliaryFeatureSet_)
            
            elif self.verbsFeature and token["tags"][0] == VERB:
                
                if token["lemma"] not in self.verbsFeatureSet_:
                    
                    self.verbsFeatureSet_[token["lemma"]] = len(self.verbsFeatureSet_)
                    
            elif self.adverbsFeature and token["tags"][0] == ADVERB:
                
                if token["lemma"] not in self.adverbsFeatureSet_:
                    
                    self.adverbsFeatureSet_[token["lemma"]] = len(self.adverbsFeatureSet_)
    # Input: proposition/sentence 
    # Output: all tokens belonging to the desired categories
    # This function is relevant for the construction of the Term-Document Matrix)
    def getAllCombinationsOfTokens(self, tokens):
        
        if self.cleanCorpus:
            tokens= self.removeStopWords(tokens)
        
        verbsList= []
        adverbsList= []
        modalsList= []
        
        
        for token in tokens:
            
            if self.verbsFeature:
                
                if token["tags"][0] == VERB:
                    verbsList.append(token["lemma"])
                    
            
            if self.adverbsFeature:
                
                if token["tags"][0] == ADVERB:
                    adverbsList.append(token["lemma"])
            
            
            if self.modalAuxiliaryFeature:
                
                if token["lemma"] in self.modalAuxiliaryList:
                    modalsList.append(token["lemma"])
            
            
        
        return modalsList + verbsList + adverbsList

    def removeStopWords(self, tokensList):
        
        punctuationMarks= string.punctuation
        punctuationMarksList= [punctuationMarks[i] for i in xrange(len(punctuationMarks))]
        
        posTagProperNoun= ['NNP', 'NNPS']
        
        stopWordsFile= open(parameters.paths["keywords_en"] + '/' + parameters.filenames["stopwords"],'r')
        
        stopWordsList= []
        
        for word in stopWordsFile:
            stopWordsList.append(word.rstrip("\n").decode("utf-8"))
        
        # list of tokens after removing: punctuation marks, stop words, proper nouns and numbers
        tokensList = [t for t in tokensList if not ( (t["lemma"] in punctuationMarksList) or (t["lemma"] in stopWordsList ) or ( t["tags"][0] in posTagProperNoun ) or (t["tags"] == 'CD' ) )]
        
        return tokensList

    def fit(self, X, y=None, **fit_params):
        return self
    
    #def get_feature_names(self):
    def vocabulary_(self):
        return self.vocabulary_ #.keys()
    
    def get_content(self):
        return self.termDocumentMatrix
    
    # override method from BaseEstimator class
    # the parameter self.modalAuxiliaryList should be updated according to updates in the parameter "self.modalAuxiliaryFeature"
    def set_params(self, **params):
        
        
        modalAuxiliaryBooleanValueBeforeUpdate= self.modalAuxiliaryFeature
        
        # call inherited method from BaseEstimator class to update some of the parameters
        super(SyntacticalTransformer, self).set_params(**params)
        
        
        
        if self.modalAuxiliaryFeature and (not modalAuxiliaryBooleanValueBeforeUpdate):
            
            self.modalAuxiliaryList= []
            
            modalAuxiliariesFile= open(parameters.paths["keywords_en"] + '/' + parameters.filenames["modalAuxiliary"],'r')
            
            
            
            for word in modalAuxiliariesFile:
                (self.modalAuxiliaryList).append(word.rstrip("\n"))
                
        elif ( not self.modalAuxiliaryFeature) and modalAuxiliaryBooleanValueBeforeUpdate:
            self.modalAuxiliaryList= []
