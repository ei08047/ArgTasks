#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ConfigurationASD:
"""

#from sklearn.neural_network import MLPClassifier


#from polyglot.mapping import Embedding
# Paths
# from polyglot.mapping import Embedding

# Paths
# from polyglot.mapping import Embedding
# Paths
# from polyglot.mapping import Embedding
import time


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC

import postagger.EnglishTagger as EnglishTagger
import utils.WordNet as WordNet
from pcd_en.FeatureAnalysis import FeatureAnalysis
from pcd_en.ml.Configuration import Configuration
# Paths
from utils.Parameters import Parameters

# from sklearn.neural_network import MLPClassifier
parameters= Parameters()

class ConfigurationPCD(Configuration):

    def loadFeatureSet(self):

        if self.type == "grid":
            return FeatureUnion([
                          ],)
        elif self.type == "n_gram":
            return FeatureUnion([
                ('ngram', CountVectorizer(tokenizer=EnglishTagger.myTokenizer, analyzer="word",
                                          preprocessor=EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary=True,
                                          stop_words="english")),
            ],)
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFeatureSet()!")
    def loadFeaturesConfigs(self):

        if self.type == "grid":
            return {
                'features__verbType__featureSetConfiguration': (0,1,2),
                'features__arguing_lexicon__featureSetConfiguration': (1,),
                'features__subjectivity_lexicon__featureSetConfiguration': (1,),

            }
        elif self.type == "n_gram":
            return {
                'features__ngram__max_features': (1000),
            }
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFeaturesConfig()!")
    def loadClassifiersConfigs(self):
        classifiers = []

        if self.type == "grid":
            ### Support Vector Machine ###
            print('grid style loadClassifiersConfigs')
            svm = SVC()
            svmParameters = {'clf__kernel': ('linear',), 'clf__probability': (True,), 'clf__class_weight': ('balanced',)}
            classifiers.append(['svm', svm, svmParameters])
        elif self.type != "grid":
            print('standard loadClassifiersConfigs')
            ### Support Vector Machine ###
            svm = SVC()
            svmParameters = {'clf__kernel': ('linear'), 'clf__probability': (True), 'clf__class_weight': ('balanced')}
            classifiers.append(['svm', svm, svmParameters])
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadClassifierConfigs()!")
        return classifiers
        
        return classifiers
    def loadFilterMethodsConfigs(self):
        filterMethods = []
        if self.type == "grid":
            ### None Feature Analysis ###
            # Represents the possibility to don't perform feature analysis
            noneFA = FeatureAnalysis()
            # Note: There are more possible parameters to tune in this algorithm
            noneFAParameters = {}
            filterMethods.append(['noneFA', noneFA, noneFAParameters])
        elif self.type != "grid":
            ### None Feature Analysis ###
            # Represents the possibility to don't perform feature analysis
            noneFA = FeatureAnalysis()
            # Note: There are more possible parameters to tune in this algorithm
            noneFAParameters = {}
            filterMethods.append(['noneFA', noneFA, noneFAParameters])
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFilterMethodsConfigs()!")
        return filterMethods
    def loadExternalTools(self):
        ###   Load external tools   ###
        
        # get ContoPt
        wordnetLoadTimeStart= time.time()
        
        wordnet= WordNet.WordNetInterface()
        
        elapsedTimeWordnetLoad= time.time() - wordnetLoadTimeStart
        print ("\nWordnet loaded in " + str(elapsedTimeWordnetLoad) + " sec.]\n")
        
        # get word2vec model
        wordEmbeddingLoadTimeStart= time.time()
        
        wordEmbeddingsModel = Embedding.load(parameters.paths["wordEmbeddings"] + "/" + parameters.filenames["wordEmbeddingsModel_en"])
        #wordEmbeddingsModel = wordEmbeddingsModel.normalize_words()
        
        elapsedTimeWordEmbeddingLoad= time.time() - wordEmbeddingLoadTimeStart
        print ("\nWord2vec model loaded in " + str(elapsedTimeWordEmbeddingLoad) + " sec.]\n")
        
        return (wordnet, wordEmbeddingsModel)
