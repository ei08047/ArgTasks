#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DatasetLoaderASD: build dataset for Argumentative Sentence Detection
"""
import math

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from pymongo import MongoClient
from sklearn.datasets.base import Bunch
from sklearn.utils import shuffle

import asd_en.BratToolReader as BratToolReader
import utils.Util as Util
from asd_en.ml.DatasetLoader import DatasetLoader
from utils.Parameters import Parameters

parameters= Parameters()
paths= parameters.paths


class DatasetLoaderASD(DatasetLoader):

    def addLearningInstancesToDataset(self):
        print('starting addLearningInstancesToDataset')
        # connect to db
        mongoClient = MongoClient('localhost', 27017) # ,connectTimeoutMS=10000000
        aaecCorpus = mongoClient.AAECCorpus
        # Sentence's table
        articleCollection= aaecCorpus.article
        # delete all elements from collections before inserting new elements
        #articleCollection.drop()
        # Sentence's table
        sentenceCollection= aaecCorpus.sentence
        # delete all elements from collections before inserting new elements
        #sentenceCollection.drop()
        if (articleCollection.find({}).count() > 0) and (sentenceCollection.find({}).count() > 0):
            # create BratToolReader object to retrieve all the Argument Diagrams from a corpus annotated using the BratNLP Tool
            bratToolReader= BratToolReader.BratToolReader()
            aaecCorpusAnnotationFilenames= bratToolReader.getAllFilenamesFromDir(paths["AAECCorpus"], ".txt")
            for filename in aaecCorpusAnnotationFilenames:
                #print('filename ' , filename)
                currentArgumentDiagram= bratToolReader.getArgumentDiagram(filename)
                currentArticleSentences= sentenceCollection.find({"articleId": int(currentArgumentDiagram.newsId)}).sort([("sentenceId", 1)])
                for sentence in currentArticleSentences:
                    # add sentence to dataset, "data" field
                    #print('sentence ', sentence["sentenceId"])
                    ((self.dataset).data).append((int(currentArgumentDiagram.newsId), int(sentence["sentenceId"])))
                    
                    # add sentence to dataset, "target" field
                    if self.determineSentenceTargetValue(sentence["originalText"], currentArgumentDiagram.graph):
                        ((self.dataset).target).append(1)
                    else:
                        ((self.dataset).target).append(0)
        else:
            # create BratToolReader object to retrieve all the Argument Diagrams from a corpus annotated using the BratNLP Tool
            bratToolReader= BratToolReader.BratToolReader()
            
            wordnet_lemmatizer = WordNetLemmatizer()
            wordPunctTokenizer= WordPunctTokenizer()
            stanfordPoSTagger= StanfordPOSTagger(paths["stanfordPoSTaggerModel"], paths["stanfordPoSTaggerJar"], encoding= "utf-8")
            
            aaecCorpusAnnotationFilenames= bratToolReader.getAllFilenamesFromDir(paths["AAECCorpus"], ".txt")
            
            
            for filename in aaecCorpusAnnotationFilenames:
                
                print (">>> Adding article " + str(filename))
                
                currentArgumentDiagram= bratToolReader.getArgumentDiagram(filename)
                
                # add news to mongo db
                if articleCollection.find({"_id": int(currentArgumentDiagram.newsId)}).count() == 0:
                    currentArticle= {
                        "_id": int(currentArgumentDiagram.newsId),
                        "body": currentArgumentDiagram.originalText
                        }
                    
                    articleCollection.insert_one(currentArticle)
                    
                    
                    sentenceId= 0
                    
                    # get sentences from current article body and corresponding tokens
                    for sentence in sent_tokenize((currentArgumentDiagram.originalText)):
                        
                        # PoS tags for current sentence
                        tokenSet= wordPunctTokenizer.tokenize(sentence)
                        tokensPoS= stanfordPoSTagger.tag(tokenSet)
                        
                        lemmas= []
                        
                        for tok in tokensPoS:
                            if treebankToWordnetPOSTags(tok[1]) == '':
                                lemmas.append(wordnet_lemmatizer.lemmatize(tok[0]))
                            else:
                                lemmas.append(wordnet_lemmatizer.lemmatize(tok[0], pos= treebankToWordnetPOSTags(tok[1])))
                        
                        sentenceTokens= [{"content": tokensPoS[tokIndex][0], "lemma": lemmas[tokIndex], "tags": tokensPoS[tokIndex][1]} for tokIndex in range(len(tokensPoS))]
                        
                        # add sentence to db
                        currentSentence= {
                            # "sentenceId" is the absolute position of the sentence in the news article
                            "sentenceId": sentenceId,
                            # "articleId" is a foreign key to the article id where this sentence was extracted from
                            "articleId": int(currentArgumentDiagram.newsId),
                            # original text extracted from the article body
                            "originalText": sentence,
                            # set of tokens, where each token has the following format: (content, lemma, tags)
                            "tokens": sentenceTokens
                            }
                        
                        sentenceCollection.insert_one(currentSentence)
                        
                        
                        # add sentence to dataset, "data" field
                        ((self.dataset).data).append((currentArgumentDiagram.newsId, sentenceId))
                        
                        # add sentence to dataset, "target" field
                        if self.determineSentenceTargetValue(sentence, currentArgumentDiagram.graph):
                            ((self.dataset).target).append(1)
                        else:
                            ((self.dataset).target).append(0)
                        
                        
                        sentenceId += 1
                        
                    
                else:
                    print ("[Warning] This Article already exists in the db!")
        # add target names to dataset bunch
        (self.dataset).target_names= ['no argument', 'argument']
        # close connection to database
        mongoClient.close()
        print('ended addLearningInstancesToDataset')

        # get Textual Entailment predictions for each sentence in the dataset
        # self.getTextualEntailmentPredictions((self.dataset).data)

    # output: bunch of the training set divided by percentage of articles "self.trainingSetPercentage" to include in the training set
    def getTrainingTestSetSplit(self, trainingSetPercentageSplit= 0.8, randomStateSeed= 12345):
        print('starting getTrainingTestSetSplit')
        trainingSet= Bunch()
        testSet= Bunch()
        articleIdsSet= shuffle(list(set([elem[0] for elem in (self.dataset).data])), random_state= randomStateSeed)
        articlesIdsTestSet= articleIdsSet[0:int(math.floor(( ((1.0 - trainingSetPercentageSplit) * len(articleIdsSet)) - 1)))]
        trainingSet.data= [elem for elem in (self.dataset).data if elem[0] not in articlesIdsTestSet]
        trainingSet.target= [(self.dataset).target[elemIndex] for elemIndex in range(len((self.dataset).target)) if (self.dataset).data[elemIndex][0] not in articlesIdsTestSet]
        trainingSet.target_names= (self.dataset).target_names
        testSet.data= [elem for elem in (self.dataset).data if elem[0] in articlesIdsTestSet]
        testSet.target= [(self.dataset).target[elemIndex] for elemIndex in range(len((self.dataset).target)) if (self.dataset).data[elemIndex][0] in articlesIdsTestSet]
        testSet.target_names= (self.dataset).target_names
        print('ended getTrainingTestSetSplit')
        return (trainingSet, testSet)

    def determineSentenceTargetValue(self, sentence, annotation):
        currentSentenceIsArgument= False
        
        for node in annotation.nodes:
            if ((node.nodeInfo).nodeType == 'MajorClaim') or ((node.nodeInfo).nodeType == 'Claim') or ((node.nodeInfo).nodeType == 'Premise'):
                #TODO: there is a build in function that is faster and may do the same thing
                if Util.relaxedSubStringSimilarity((node.nodeInfo).text, sentence, 0.85):
                    currentSentenceIsArgument= True
                    break
        
        return currentSentenceIsArgument

    def getDatasetInfo(self):
        numberOfSentences= 0
        numberOfArgumentativeSentences= 0
        for i in range(0, len((self.dataset).data)):
            numberOfSentences += 1
            if (self.dataset).target[i] == 1:
                numberOfArgumentativeSentences += 1
        
        print ("no. argumentative sentences= " + str(numberOfArgumentativeSentences))
        print ("no. not argumentative sentences= " + str(numberOfSentences - numberOfArgumentativeSentences))
        print ("total no. of sentences= " + str(numberOfSentences))

def treebankToWordnetPOSTags(treebank_tag):
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

"""
### Test ###
myDatasetLoader = DatasetLoaderASD()
#print myDatasetLoader
myDatasetLoader.getDatasetInfo()
"""


