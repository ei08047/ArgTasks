from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
import string
from utils.Parameters import Parameters
import re
import numpy as np
from pymongo import MongoClient
parameters = Parameters()
from nltk.tokenize import word_tokenize

"""
configs:
0- No feature
1- sentence polarity TODO:(needs parameterization )
2- num_polarity_words / num_neutral_words
3- mean_pos mean_neg
4- median_pos median_neg
"""

class Tagger:
    def myTokenizer(proposition):

        tokens = proposition['tokens']
        relevant_tokens = [token for token in tokens if not token.isdigit() and token not in string.punctuation]
        #print('tokens: ', tokens)
        return relevant_tokens

    def myPreProcessor(learningInstance):
        # connect to db
        mongoClient = MongoClient('localhost', 27017)

        ffCorpus = mongoClient.FACTFEELCorpus
        documentCollection = ffCorpus.documents
        ret = documentCollection.find_one({'document_id': learningInstance })
        mongoClient.close()
        return ret







