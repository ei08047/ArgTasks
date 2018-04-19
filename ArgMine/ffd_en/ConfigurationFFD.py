#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ConfigurationFFD:
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC

from ffd_en.FFTagger import Tagger
from ffd_en.FeatureAnalysis import FeatureAnalysis
from ffd_en.Transformers.AfinnTransformer import AfinnTransformer
from ffd_en.Transformers.ArguingLexiconTransformer import ArguingLexiconTransformer
from ffd_en.Transformers.SubjectivityLexiconTransformer import SubjectivityLexiconTransformer
from ffd_en.Transformers.VaderTransformer import VaderTransformer
from ffd_en.Transformers.VerbTypeTransformer import VerbTypeTransformer
from ffd_en.Transformers.NumberTransformer import NumberTransformer
from ffd_en.ml.Configuration import Configuration
from utils.Parameters import Parameters

parameters = Parameters()


class ConfigurationFFD(Configuration):

    def loadFeatureSet(self):

        if self.type == "baseline":
            return FeatureUnion([
                ('verbType', VerbTypeTransformer()),
                ('subjectivity_lexicon', SubjectivityLexiconTransformer()),
                ('afinn_lexicon', AfinnTransformer()),
                ('vader_lexicon', VaderTransformer()),
            ],
            )
        elif self.type == "grid":
            return FeatureUnion([
                ('verbType', VerbTypeTransformer()),
                ('subjectivity_lexicon', SubjectivityLexiconTransformer()),
                ('afinn_lexicon', AfinnTransformer()),
                ('vader_lexicon', VaderTransformer()),
                ('arguing_lexicon', ArguingLexiconTransformer()),
                ('numbers', NumberTransformer()),
            ],
            )
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
        elif self.type == "afinn_lexicon":
            return FeatureUnion([
                  ('afinn_lexicon', AfinnTransformer()),
            ],
            )
        elif self.type == "vader_lexicon":
            return FeatureUnion([
                  ('vader_lexicon', VaderTransformer()),
            ],
            )
        elif self.type == "one_gram": # , max_features=500
            print('loadFeatureSet one_gram' )
            return FeatureUnion([
                ('ngram', CountVectorizer(tokenizer=Tagger.myTokenizer, analyzer="word",
                                          preprocessor=Tagger.myPreProcessor, ngram_range=(1, 1), binary=True,
                                          stop_words="english"))
            ],
            )

        elif self.type == "numbers": # TODO
            return FeatureUnion([
                ('numbers', NumberTransformer()),
            ],
            )
        # TODO : fact and feel patterns
        elif self.type == "pos_patterns":
            return FeatureUnion([
                ('pos_patterns', SubjectivityLexiconTransformer()),
            ],
            )
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFeatureSet()!")

    def loadFeaturesConfigs(self):

        if self.type == "baseline":
            return {
                'features__verbType__featureSetConfiguration': (1),
                'features__subjectivity_lexicon__featureSetConfiguration': (1),
                'features__afinn_lexicon__featureSetConfiguration': (1),
                'features__vader_lexicon__featureSetConfiguration': (1),
            }
        elif self.type == "grid":
            return {
                'features__verbType__featureSetConfiguration': (0,1),
                'features__subjectivity_lexicon__featureSetConfiguration': (0,1),
                'features__afinn_lexicon__featureSetConfiguration': (0,1,2),
                'features__vader_lexicon__featureSetConfiguration': (0,1),
                'features__arguing_lexicon__featureSetConfiguration': (0,1),
                'features__numbers__featureSetConfiguration': (0,3),
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
        elif self.type == "afinn_lexicon":
            return {
                    'features__afinn_lexicon__featureSetConfiguration': (2),
            }
        elif self.type == "vader_lexicon":
            return {
                    'features__vader_lexicon__featureSetConfiguration': (1),
            }
        elif self.type == "arguing_lexicon":
            return {
                    'features__arguing_lexicon__featureSetConfiguration': (1),
            }
        elif self.type == "one_gram":
            return {
                'features__ngram__max_features': (500),
            }
        elif self.type == "numbers":
            return {
                'features__numbers__featureSetConfiguration': (3),
            }
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFeaturesConfig()!")

    def loadClassifiersConfigs(self):
        # Structure: [name, object, parameters]
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
        elif  self.type != "baseline" and self.type != "grid":
            print('standard loadClassifiersConfigs')
            ### Support Vector Machine ###
            svm = SVC()
            svmParameters = {'clf__kernel': ('linear'), 'clf__probability': (True), 'clf__class_weight': ('balanced')}
            classifiers.append(['svm', svm, svmParameters])
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadClassifierConfigs()!")
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

