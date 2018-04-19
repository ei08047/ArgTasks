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

from scipy.stats import expon
from sklearn.decomposition import PCA, NMF, FactorAnalysis, RandomizedPCA, KernelPCA, MiniBatchSparsePCA, TruncatedSVD, \
    LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif, chi2, GenericUnivariateSelect
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import postagger.EnglishTagger as EnglishTagger
import utils.WordNet as WordNet
from asd_en.FeatureAnalysis import FeatureAnalysis
from asd_en.Transformers.AfinnTransformer import AfinnTransformer
from asd_en.Transformers.ArguingLexiconTransformer import ArguingLexiconTransformer
from asd_en.Transformers.KeywordTransformer import KeywordTransformer
from asd_en.Transformers.RabbitRuleTransformer import RabbitRuleTransformer
from asd_en.Transformers.SemanticTransformer import SemanticTransformer
from asd_en.Transformers.SubjectivityLexiconTransformer import SubjectivityLexiconTransformer
from asd_en.Transformers.SyntacticNumTransformer import SyntacticNumTransformer
from asd_en.Transformers.TextStatisticsTransformer import TextStatisticsTransformer
from asd_en.Transformers.VerbTenseTransformer import VerbTenseTransformer
from asd_en.Transformers.VerbTypeTransformer import VerbTypeTransformer
from asd_en.Transformers.WordCouplesTransformer import WordCouplesTransformer
from asd_en.ml.Configuration import Configuration
# Paths
from utils.Parameters import Parameters

# from sklearn.neural_network import MLPClassifier
parameters= Parameters()


class ConfigurationASD(Configuration):

    def loadFeatureSet(self):
        if self.type == "best_20171217":
            # Load external tools
            #wordnet, wordEmbeddingsModel = self.loadExternalTools()
            return FeatureUnion([
                          ('ngram', CountVectorizer(tokenizer= EnglishTagger.myTokenizer, analyzer="word", preprocessor= EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary= True, stop_words= "english")),
                          ('syntacticNum', SyntacticNumTransformer()),
                          ('keywordsmodel', KeywordTransformer()),
                          ('textStatistics', TextStatisticsTransformer()),
                          ('verbTense', VerbTenseTransformer()),

                          ('subjectivity_lexicon', SubjectivityLexiconTransformer()),
                          ('arguing_lexicon', ArguingLexiconTransformer()),
                          #('semanticLevel', SemanticTransformer(wordnet, wordEmbeddingsModel))
                          ],
                         )
        elif self.type == "all":
            
            # Load external tools
            wordnet, wordEmbeddingsModel = self.loadExternalTools()
            
            return FeatureUnion([
                          ('ngram', CountVectorizer(tokenizer= EnglishTagger.myTokenizer, analyzer="word", preprocessor= EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary= True, stop_words= "english")),
                          ('wordCouple', WordCouplesTransformer()),
                          ('syntacticNum', SyntacticNumTransformer()),
                          ('keywordsmodel', KeywordTransformer()),
                          ('textStatistics', TextStatisticsTransformer()),
                          ('verbTense', VerbTenseTransformer()),
                          ('rabbitRule', RabbitRuleTransformer(wordnet, wordEmbeddingsModel)),
                          ('semanticLevel', SemanticTransformer(wordnet, wordEmbeddingsModel))
                          ],
                         )
        elif self.type == "fast":
            return FeatureUnion([
                          ('syntacticNum', SyntacticNumTransformer()),
                          ('keywordsmodel', KeywordTransformer()),
                          ('textStatistics', TextStatisticsTransformer()),
                          ('verbTense', VerbTenseTransformer()),
                          ],
                         )
        elif self.type == "grid":
            return FeatureUnion([
                ('subjectivity_lexicon', SubjectivityLexiconTransformer()),
                ('arguing_lexicon', ArguingLexiconTransformer()),
                ('verbType', VerbTypeTransformer()),
                          ],
                         )
        elif self.type == "n_gram":
            return FeatureUnion([
                ('ngram', CountVectorizer(tokenizer=EnglishTagger.myTokenizer, analyzer="word",
                                          preprocessor=EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary=True,
                                          stop_words="english")),
            ],)
        elif self.type == "verb_lexicon":
            return FeatureUnion([
                  ('verbType', VerbTypeTransformer()),
            ],
            )
        elif self.type == "subjectivity_lexicon":
            return FeatureUnion([
                  ('subjectivity_lexicon', SubjectivityLexiconTransformer()),
            ],
            )
        elif self.type == "subjectivity":
            return FeatureUnion([
                    ('subjectivity_lexicon', SubjectivityLexiconTransformer()),
                    ('verbType', VerbTypeTransformer()),
            ],
            )
        elif self.type == "arguing_lexicon":
            return FeatureUnion([
                  ('arguing_lexicon', ArguingLexiconTransformer()),
            ],
            )
        elif self.type == "baseline":
            return FeatureUnion([
                          ('ngram', CountVectorizer(tokenizer= EnglishTagger.myTokenizer, analyzer="word", preprocessor= EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary= True, stop_words= "english")),
                          ('syntacticNum', SyntacticNumTransformer()),
                          ('keywordsmodel', KeywordTransformer()),
                          ('textStatistics', TextStatisticsTransformer()),
                          ('verbTense', VerbTenseTransformer()),
                          ],
                         )
        elif self.type == "baseline_subjectivity_lexicon":
            return FeatureUnion([
                          ('ngram', CountVectorizer(tokenizer=EnglishTagger.myTokenizer, analyzer="word",preprocessor=EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary=True,stop_words="english")),
                          ('syntacticNum', SyntacticNumTransformer()),
                          ('keywordsmodel', KeywordTransformer()),
                          ('textStatistics', TextStatisticsTransformer()),
                          ('verbTense', VerbTenseTransformer()),
                          ('subjectivity_lexicon', SubjectivityLexiconTransformer()),
            ],
                         )
        elif self.type == "baseline_arguing_lexicon":
            return FeatureUnion([
                  ('ngram', CountVectorizer(tokenizer=EnglishTagger.myTokenizer, analyzer="word",preprocessor=EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary=True,stop_words="english")),
                  ('syntacticNum', SyntacticNumTransformer()),
                  ('keywordsmodel', KeywordTransformer()),
                  ('textStatistics', TextStatisticsTransformer()),
                  ('verbTense', VerbTenseTransformer()),
                  ('arguing_lexicon', ArguingLexiconTransformer()),
            ],
                         )
        elif self.type == "baseline_verb_lexicon":
            return FeatureUnion([
                  ('ngram', CountVectorizer(tokenizer=EnglishTagger.myTokenizer, analyzer="word",preprocessor=EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary=True,stop_words="english")),
                  ('syntacticNum', SyntacticNumTransformer()),
                  ('keywordsmodel', KeywordTransformer()),
                  ('textStatistics', TextStatisticsTransformer()),
                  ('verbTense', VerbTenseTransformer()),
                  ('verbType', VerbTypeTransformer()),
            ],
                         )
        elif self.type == "baseline_arguing_lexicon_subjectivity_lexicon":
            return FeatureUnion([
                  ('ngram', CountVectorizer(tokenizer=EnglishTagger.myTokenizer, analyzer="word",preprocessor=EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary=True,stop_words="english")),
                  ('syntacticNum', SyntacticNumTransformer()),
                  ('keywordsmodel', KeywordTransformer()),
                  ('textStatistics', TextStatisticsTransformer()),
                  ('verbTense', VerbTenseTransformer()),
                  ('arguing_lexicon', ArguingLexiconTransformer()),
                  ('subjectivity_lexicon', SubjectivityLexiconTransformer()),

            ],
                         )
        elif self.type == "baseline_arguing_lexicon_subjectivity_lexicon_verb_lexicon":
            return FeatureUnion([
                  ('ngram', CountVectorizer(tokenizer=EnglishTagger.myTokenizer, analyzer="word",preprocessor=EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary=True,stop_words="english")),
                  ('syntacticNum', SyntacticNumTransformer()),
                  ('keywordsmodel', KeywordTransformer()),
                  ('textStatistics', TextStatisticsTransformer()),
                  ('verbTense', VerbTenseTransformer()),
                  ('arguing_lexicon', ArguingLexiconTransformer()),
                  ('subjectivity_lexicon', SubjectivityLexiconTransformer()),
                  ('verbType', VerbTypeTransformer()),
            ],
                         )
        elif self.type == "only_emotion_transformer":
            print('returning only_emotion_transformer feature set')
            return FeatureUnion([
                ('ngram', CountVectorizer(tokenizer=EnglishTagger.myTokenizer, analyzer="word",
                                          preprocessor=EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary=True,
                                          stop_words="english")),
                ('EmotionDetection' , AfinnTransformer())
            ],
            )
        elif self.type == "baseline_verb_type":
            return FeatureUnion([
                          ('ngram', CountVectorizer(tokenizer= EnglishTagger.myTokenizer, analyzer="word", preprocessor= EnglishTagger.myPreProcessor, ngram_range=(1, 1), binary= True, stop_words= "english")),
                          ('syntacticNum', SyntacticNumTransformer()),
                          ('keywordsmodel', KeywordTransformer()),
                          ('textStatistics', TextStatisticsTransformer()),
                          ('verbTense', VerbTenseTransformer()),
                          ('verbType', VerbTypeTransformer())
                          ],
                         )
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFeatureSet()!")

    def loadFeaturesConfigs(self):
        if self.type == "best_20171217":
            return {

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
                    #'features__semanticLevel__cosineSimilarity': (1),
                    #'features__semanticLevel__currentPropositionVersor': (1),
                    'features__arguing_lexicon__featureSetConfiguration': (1),
                    'features__subjectivity_lexicon__featureSetConfiguration': (1),
                    }
        elif self.type == "all":
            return {
                    'features__wordCouple__numberOfKeywords' : (2),
                    'features__wordCouple__cleanCorpus' : (True),
                    
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
                    
                    'features__rabbitRule__featureSetConfiguration': (1),
                    
                    'features__semanticLevel__cosineSimilarity': (1),
                    'features__semanticLevel__currentPropositionVersor': (1),
                    }
        elif self.type == "fast":
            return {
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
                    }
        elif self.type == "grid":
            return {
                'features__verbType__featureSetConfiguration': (0,1,2),
                'features__arguing_lexicon__featureSetConfiguration': (1,),
                'features__subjectivity_lexicon__featureSetConfiguration': (1,),

            }
        elif self.type == "n_gram":
            return {
                'features__ngram__max_features': (1000),
            }
        elif self.type == "verb_lexicon":
            return {
                    'features__verbType__featureSetConfiguration': (1),
            }
        elif self.type == "subjectivity_lexicon":
            return {
                    'features__subjectivity_lexicon__featureSetConfiguration': (1),
            }
        elif self.type == "subjectivity":
            return {
                    'features__verbType__featureSetConfiguration': (1),
                    'features__subjectivity_lexicon__featureSetConfiguration': (1),
            }
        elif self.type == "arguing_lexicon":
            return {
                    'features__arguing_lexicon__featureSetConfiguration': (1),
            }
        elif self.type == "baseline":
            return {
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
            }
        elif self.type == "baseline_subjectivity_lexicon":
            return {
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
                    'features__subjectivity_lexicon__featureSetConfiguration': (1),
            }
        elif self.type == "baseline_arguing_lexicon":
            return {
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
                    'features__arguing_lexicon__featureSetConfiguration': (1),

            }
        elif self.type == "baseline_verb_lexicon":
            return {
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
                    'features__verbType__featureSetConfiguration': (1),
            }
        elif self.type == "baseline_arguing_lexicon_subjectivity_lexicon":
            return {
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
                    'features__arguing_lexicon__featureSetConfiguration': (1),
                    'features__subjectivity_lexicon__featureSetConfiguration': (1),
            }
        elif self.type == "baseline_arguing_lexicon_subjectivity_lexicon_verb_lexicon":
            return {
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
                    'features__arguing_lexicon__featureSetConfiguration': (1),
                    'features__subjectivity_lexicon__featureSetConfiguration': (1),
                    'features__verbType__featureSetConfiguration': (1),
            }
        elif self.type == "only_emotion_transformer":
            return {
                'features__EmotionDetection__featureSetConfiguration': (1)
            }
        elif self.type == "baseline_afinn":
            return {
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
                'features__EmotionDetection__featureSetConfiguration': (1)
            }
        elif self.type == "baseline_verb_type":
            return {
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
                'features__verbType__featureSetConfiguration': (1)
            }
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFeaturesConfig()!")

    def loadClassifiersConfigs(self):
        classifiers = []
        if self.type == "baseline":
            ### Support Vector Machine ###
            svm = SVC()
            svmParameters = {'clf__kernel': ('linear'), 'clf__probability': (True), 'clf__class_weight': ('balanced')}
            classifiers.append(['svm', svm, svmParameters])
        elif self.type == "grid":
            ### Support Vector Machine ###
            print('grid style loadClassifiersConfigs')
            svm = SVC()
            svmParameters = {'clf__kernel': ('linear',), 'clf__probability': (True,), 'clf__class_weight': ('balanced',)}
            classifiers.append(['svm', svm, svmParameters])
        elif self.type != "baseline" and self.type != "grid":
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
        if self.type == "baseline":
            ### None Feature Analysis ###
            # Represents the possibility to don't perform feature analysis
            noneFA = FeatureAnalysis()
            # Note: There are more possible parameters to tune in this algorithm
            noneFAParameters = {}
            filterMethods.append(['noneFA', noneFA, noneFAParameters])
        elif self.type != "baseline":
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
