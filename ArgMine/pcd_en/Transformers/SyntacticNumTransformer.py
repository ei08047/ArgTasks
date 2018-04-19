"""
SyntacticTransformer: Custom Transformer to create syntactic level features.
It receives as input a matrix with dimensions 1:X where each element is a sentence/proposition
(being X the number of propositions in the fold). Outputs a matrix with dimension 
X:Y, where:
Y= A + B + C,
A = set of features with information on Modal Auxiliaries (if  modalAuxiliaryFeature is True)
B = set of features with information on Verbs (if  verbsFeature is True),
C = set of features with information on Adverbs (if  adverbsFeature is True),
The feature set Y corresponds to the disjoint union of the individual feature sets A, B and C, with the features of A followed 
by the features of B and the features B followed by the features of C (features are not mixed). Each element is an integer.
"""


import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator

from pymongo import MongoClient

from utils.Parameters import Parameters
parameters= Parameters()

VERB= 'V'
ADVERB= 'R'


class SyntacticNumTransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, featureSetConfiguration= 1, modalAuxiliaryFeature= True, verbsFeature= False, adverbsFeature= True, cleanCorpus= False):
        # feature set configuration 
        # If == 0, binary features are returned
        # If == 1, total counts are returned
        if (featureSetConfiguration) >= 0 and (featureSetConfiguration < 2):
            self.featureSetConfiguration= featureSetConfiguration
        else:
            # default value
            self.featureSetConfiguration= 0
        # Boolean variable indication whether we should clean the corpus (True) or not (False)
        self.cleanCorpus= cleanCorpus
        # Boolean indicating if we are interested in verbs as features
        self.verbsFeature= verbsFeature
        # Boolean indicating if we are interested in adverbs as features
        self.adverbsFeature= adverbsFeature
        # Boolean indicating if we are interested in modal auxiliary as features
        self.modalAuxiliaryFeature= modalAuxiliaryFeature
        # List of modal auxiliary keywordsmodel -> this list corresponds to the set of words that are considered to be modal auxiliary
        # in the Portuguese language
        self.modalAuxiliaryList= []
        if self.modalAuxiliaryFeature:
            modalAuxiliariesFile= open(parameters.paths["keywords_en"] + '\\' + parameters.filenames["modalAuxiliary"],'r', encoding='utf-8')
            for word in modalAuxiliariesFile:
                (self.modalAuxiliaryList).append(word.rstrip("\n"))

    def transform(self, X, **transform_params):
        self.featureArray= []
        if (not self.modalAuxiliaryFeature) and (not self.verbsFeature) and (not self.adverbsFeature):
            self.featureArray= [[0] for j in range(len(X))]
        else:
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            aaecCorpus = mongoClient.AAECCorpus
            # Sentence's table
            sentenceCollection= aaecCorpus.sentence
            for sentenceTuple in X:
                # Feature Set for current learning instance
                currentLearningInstanceFeatureSet= []
                # current learning instance info from database
                currentSentence= sentenceCollection.find_one({"$and":[{"articleId": int(sentenceTuple[0])}, {"sentenceId": int(sentenceTuple[1])}]})
                modalAuxCount= 0
                verbCount= 0
                advCount= 0
                
                # modalAuxiliary feature
                if self.modalAuxiliaryFeature:
                    for t in currentSentence["tokens"]:
                        for modalAuxiliaryWord in self.modalAuxiliaryList:
                            # check if token lemma is modal auxiliary
                            if t["lemma"] == modalAuxiliaryWord:
                                modalAuxCount = modalAuxCount + 1
                    
                    if self.featureSetConfiguration == 0:
                        if modalAuxCount > 0:
                            currentLearningInstanceFeatureSet.append(1)
                        else:
                            currentLearningInstanceFeatureSet.append(0)
                    else:
                        currentLearningInstanceFeatureSet.append(modalAuxCount)
                # verb feature
                if self.verbsFeature:
                    for t in currentSentence["tokens"]:
                        if t["tags"][0] == VERB:
                            verbCount = verbCount + 1
                    if self.featureSetConfiguration == 0:
                        if verbCount > 0:
                            currentLearningInstanceFeatureSet.append(1)
                        else:
                            currentLearningInstanceFeatureSet.append(0)
                    else:
                        currentLearningInstanceFeatureSet.append(verbCount)
                # adverb feature
                if self.adverbsFeature:
                    for t in currentSentence["tokens"]:
                        if t["tags"][0] == ADVERB:
                            advCount = advCount + 1
                    if self.featureSetConfiguration == 0:
                        if advCount > 0:
                            currentLearningInstanceFeatureSet.append(1)
                        else:
                            currentLearningInstanceFeatureSet.append(0)
                    else:
                        currentLearningInstanceFeatureSet.append(advCount)
                (self.featureArray).append(currentLearningInstanceFeatureSet)
            # close database connection
            mongoClient.close()
        return np.asarray(self.featureArray)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_content(self):
        return np.asarray(self.featureArray)
    
    def get_feature_names(self):
        syntacticLevelFeaturesNames= []
        if (not self.modalAuxiliaryFeature) and (not self.verbsFeature) and (not self.adverbsFeature):
            return ['None']
        if self.modalAuxiliaryFeature:
            if self.featureSetConfiguration == 0:
                syntacticLevelFeaturesNames.append("modalAuxiliaryInProposition")
            elif self.featureSetConfiguration == 1:
                syntacticLevelFeaturesNames.append("modalAuxiliaryCount")
        if self.verbsFeature:
            if self.featureSetConfiguration == 0:
                syntacticLevelFeaturesNames.append("verbsInProposition")
            elif self.featureSetConfiguration == 1:
                syntacticLevelFeaturesNames.append("verbsCount")
        if self.adverbsFeature:
            if self.featureSetConfiguration == 0:
                syntacticLevelFeaturesNames.append("adverbsInProposition")
            elif self.featureSetConfiguration == 1:
                syntacticLevelFeaturesNames.append("adverbsCount")
        return syntacticLevelFeaturesNames
