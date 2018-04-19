#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ConfigurationASD:
"""

import time

import FeatureAnalysis
import KeywordTransformer
import SemanticTransformer
import SyntacticNumTransformer
import TextStatisticsTransformer
import VerbTenseTransformer
import WordCouplesTransformer
from polyglot.mapping import Embedding
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC

import postagger.CitiusTagger as POSTagger
import utils.ContoPtReader as ContoPtReader
from asd_en.ml import Configuration
# Paths
from utils.Parameters import Parameters

# from sklearn.neural_network import MLPClassifier
parameters= Parameters()


class ConfigurationASD(Configuration):

    def loadFeatureSet(self):
        if self.type == "best":
            # Load external tools
            wordnet, wordEmbeddingsModel = self.loadExternalTools()
            return FeatureUnion([
                          ('ngram', CountVectorizer(tokenizer= POSTagger.myTokenizer, analyzer="word", preprocessor= POSTagger.myPreProcessor, ngram_range=(1, 1), binary= True)),
                          ('wordCouple', WordCouplesTransformer.WordCouplesTransformer()),
                          ('syntacticNum', SyntacticNumTransformer.SyntacticNumTransformer()),
                          ('keywordsmodel', KeywordTransformer.KeywordTransformer()),
                          ('textStatistics', TextStatisticsTransformer.TextStatisticsTransformer()),
                          ('verbTense', VerbTenseTransformer.VerbTenseTransformer()),
                          ('semanticLevel', SemanticTransformer.SemanticTransformer(wordnet, wordEmbeddingsModel))
                          ],
                         )
        elif self.type == "fast":
            return FeatureUnion([
                          ('wordCouple', WordCouplesTransformer.WordCouplesTransformer()),
                          ('syntacticNum', SyntacticNumTransformer.SyntacticNumTransformer()),
                          ],
                         )
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFeatureSet()!")

    def loadFeaturesConfigs(self):
        
        if self.type == "best":
            return {
                    'features__wordCouple__numberOFKeywords' : (2),
                    'features__wordCouple__cleanCorpus' : (True),
                    'features__wordCouple__active' : (True),
                    
                    'features__syntacticNum__featureSetConfiguration': (0),
                    'features__syntacticNum__modalAuxiliaryFeature': (True),
                    'features__syntacticNum__verbsFeature': (False),
                    'features__syntacticNum__adverbsFeature': (True),
                    'features__syntacticNum__cleanCorpus': (False),
                    
                    'features__keywordsmodel__featureSetConfiguration': (2),
                    
                    'features__textStatistics__sentenceLength': (False),
                    'features__textStatistics__averageWordLength': (False),
                    'features__textStatistics__punctuationMarks': (False),
                    'features__textStatistics__absolutePosition': (True),
                    
                    'features__verbTense__featureSetConfiguration': (3),
                    
                    'features__semanticLevel__cosineSimilarity': (1),
                    'features__semanticLevel__currentPropositionVersor': (1),
                    }
        
            
        elif self.type == "fast":
            return {
                    'features__wordCouple__numberOFKeywords' : (2),
                    'features__wordCouple__cleanCorpus' : (True),
                    'features__wordCouple__active' : (True),
                    
                    'features__syntacticNum__featureSetConfiguration': (0),
                    'features__syntacticNum__modalAuxiliaryFeature': (True),
                    'features__syntacticNum__verbsFeature': (False),
                    'features__syntacticNum__adverbsFeature': (True),
                    'features__syntacticNum__cleanCorpus': (False),
                    }
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFeaturesConfig()!")

    def loadClassifiersConfigs(self):
        # Structure: [name, object, parameters]
        classifiers= []
        
        if self.type == "best":
            """
            ### Logistic Regression Classifier ###
            lr= LogisticRegression()
            
            lrParameters= {'clf__class_weight': ('balanced')}
            
            classifiers.append(['lr', lr, lrParameters])
            """
            ### Support Vector Machine ###
            svm= SVC()
            
            svmParameters= {'clf__kernel': ('linear'), 'clf__probability': (True), 'clf__class_weight': ('balanced')}
            
            classifiers.append(['svm', svm, svmParameters])
            
            
        elif self.type == "fast":
            ### Support Vector Machine ###
            svm= SVC()
            
            svmParameters= {'clf__kernel': ('linear'), 'clf__probability': (True), 'clf__class_weight': ('balanced')}
            
            classifiers.append(['svm', svm, svmParameters])
            
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadClassifierConfigs()!")
        
        return classifiers

    def loadFilterMethodsConfigs(self):
        filterMethods= []
        
        if self.type == "best":
            
            ### None Feature Analysis ###
            # Represents the possibility to don't perform feature analysis
            
            noneFA= FeatureAnalysis.FeatureAnalysis()
            
            # Note: There are more possible parameters to tune in this algorithm
            noneFAParameters= {}
            
            filterMethods.append(['noneFA', noneFA, noneFAParameters])
            
        
            
        elif self.type == "fast":
            
            ### None Feature Analysis ###
            # Represents the possibility to don't perform feature analysis
            
            noneFA= FeatureAnalysis.FeatureAnalysis()
            
            # Note: There are more possible parameters to tune in this algorithm
            noneFAParameters= {}
            
            filterMethods.append(['noneFA', noneFA, noneFAParameters])
            
            
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFilterMethodsConfigs()!")
        
        return filterMethods

    def loadExternalTools(self):
        ###   Load external tools   ###
        
        # get ContoPt
        wordnetLoadTimeStart= time.time()
        
        wordnet= ContoPtReader.ContoPtLoader()
        
        elapsedTimeWordnetLoad= time.time() - wordnetLoadTimeStart
        print "\nWordnet loaded in " + str(elapsedTimeWordnetLoad) + " sec.]\n"
        
        #  get word2vec model
        wordEmbeddingLoadTimeStart= time.time()
        
        wordEmbeddingsModel = Embedding.load(parameters.paths["wordEmbeddings"] + "/polyglot-pt.pkl")
        #wordEmbeddingsModel = (self.wordEmbeddingsModel).normalize_words()
        
        elapsedTimeWordEmbeddingLoad= time.time() - wordEmbeddingLoadTimeStart
        print "\nWord2vec model loaded in " + str(elapsedTimeWordEmbeddingLoad) + " sec.]\n"
        
        return (wordnet, wordEmbeddingsModel)
    
    
