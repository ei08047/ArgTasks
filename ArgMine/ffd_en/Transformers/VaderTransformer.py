from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
from pymongo import MongoClient
parameters = Parameters()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

vader = SentimentIntensityAnalyzer()

"""
configs:
0- No feature
1- ful text sentiment 
2- num_polarity_words / num_neutral_words td
3- mean_pos mean_neg td 
4- median_pos median_neg td
"""


#positive sentiment: compound score >= 0.05
#neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
#negative sentiment: compound score <= -0.05

class VaderTransformer(TransformerMixin, BaseEstimator):
    features = {'neu': 0,'compound':0,'neg':0,'pos':0 } # 'positive':0, 'neutral':0
    def __init__(self, featureSetConfiguration = 1 ):

        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):
        mongoClient = MongoClient('localhost', 27017)
        ffCorpus = mongoClient.FACTFEELCorpus
        documentCollection = ffCorpus.documents
        temp = []

        if(self.featureSetConfiguration == 0): # not active
            temp = [
                [0 for f in VaderTransformer.features.keys()]
                for s in X
            ]
        # document_score
        elif self.featureSetConfiguration == 1:
            for document in X:
                features_to_set = {'vader_value': 0}
                # current learning instance info from database
                currentDocument = documentCollection.find_one({'document_id': document })
                raw_sentence = currentDocument['raw'].lower()
                vader_result = vader.polarity_scores(raw_sentence)
                b = vader_result
                features_to_set['compound'] = b['compound']
                features_to_set['pos'] = b['pos']
                features_to_set['neg'] = b['neg']
                features_to_set['neu'] = b['neu']
                temp.append([features_to_set[key] for key in sorted(features_to_set.keys())])

        # num_polarity_words / num_neutral_words
        #TODO: 2
        elif self.featureSetConfiguration == 2:
            document = ffCorpus.documents
            temp = []
            for sentenceTuple in X:
                features_to_set = {'polar/neutral': 0}
        # mean_pos mean_neg
        # TODO: 3
        elif self.featureSetConfiguration == 3:

            documentCollection = ffCorpus.documents
            temp = []
            for document in X:
                features_to_set = {'polar/neutral': 0}
        # TODO: 4
        # median_pos median_neg
        elif self.featureSetConfiguration == 4:
            documentCollection = ffCorpus.documents
            temp = []
            for document in X:
                features_to_set = {'polar/neutral': 0}

        mongoClient.close()
        features = np.array(temp)
        #print('VaderTransformer:' , self.featureSetConfiguration,' ### X:',len(X),'len(features):',len(features))
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    ## names are related to featureSetConfiguration
    def get_feature_names(self):
        return sorted(VaderTransformer.features.keys())






