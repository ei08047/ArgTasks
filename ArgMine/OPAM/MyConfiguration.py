from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC

from OPAM.Transformers.ArguingLexiconTransformer import ArguingLexiconTransformer
from OPAM.Transformers.MyFeatureAnalysis import MyFeatureAnalysis
from OPAM.Transformers.OpinionFinderTransformer import OpinionFinderTransformer
from OPAM.Transformers.SubjectivityLexiconTransformer import SubjectivityLexiconTransformer
from OPAM.Transformers.VerbTypeTransformer import VerbTypeTransformer
from asd_en.ml import Configuration


class MyConfiguration(Configuration):

    def loadFeatureSet(self):
        if self.type == "best":
            # Load external tools
            f = FeatureUnion([
                ('ngram', CountVectorizer( analyzer="word", ngram_range=(1, 1),stop_words='english', binary=True)),
                ('opinion_finder', OpinionFinderTransformer()),
                ('arguing_lexicon', ArguingLexiconTransformer()),
                ('verb_type', VerbTypeTransformer()),
                ('subjectivity_lexicon', SubjectivityLexiconTransformer())
            ])
            return f



    def loadFeaturesConfigs(self):
        if self.type == "best":
            return {
                    'features__opinion_finder__featureSetConfiguration': (1),
                    'features__arguing_lexicon__featureSetConfiguration': (1),
                    'features__verb_type__featureSetConfiguration': (1),
                    'features__subjectivity_lexicon__featureSetConfiguration': (1),
                    }

    def loadClassifiersConfigs(self):
        classifiers = []
        if self.type == "best":
            svm = SVC()
            svmParameters = {'clf__kernel': ('linear'), 'clf__probability': (True), 'clf__class_weight': ('balanced')}
            classifiers.append(['svm', svm, svmParameters])
        return classifiers

    def loadFilterMethodsConfigs(self):
        filterMethods = []
        if self.type == "best":
            noneFA = MyFeatureAnalysis()
            noneFAParameters = {}
            filterMethods.append(['noneFA', noneFA, noneFAParameters])
        elif self.type == "fast":

            ### None Feature Analysis ###
            # Represents the possibility to don't perform feature analysis

            noneFA = MyFeatureAnalysis()

            # Note: There are more possible parameters to tune in this algorithm
            noneFAParameters = {}

            filterMethods.append(['noneFA', noneFA, noneFAParameters])
        else:
            raise Exception("Invalid parameters @ ConfigurationASD.loadFilterMethodsConfigs()!")
        return filterMethods