#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EnglishTagger: set of functions to run and extract the outputs produced by the 
POS Tagger tool 
"""

from pymongo import MongoClient
import sys


# Functions

def myTokenizer(proposition):
    
    #TODO: remove stopwords here?
    return [tok["lemma"].lower() for tok in proposition["tokens"]]

def myPreProcessor(learningInstance):
    
    # connect to db
    mongoClient = MongoClient('localhost', 27017)
    
    aaecCorpus = mongoClient.AAECCorpus
    
    # Sentence's table
    sentenceCollection= aaecCorpus.sentence
    
    # return content from database
    #print('articleId  {}  sentenceId   {} type {}  sizeof {}'.format(learningInstance,learningInstance[1], type(learningInstance[0]), sys.getsizeof(learningInstance[0])))
    return sentenceCollection.find_one({"$and":[{"articleId": int(learningInstance[0])}, {"sentenceId": int(learningInstance[1])}]})

