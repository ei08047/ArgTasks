#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DatasetLoaderASD: build dataset for Argumentative Sentence Detection
"""

import codecs
import json
import time

from pymongo import MongoClient
from sklearn.datasets.base import Bunch
from sklearn.externals import joblib
from sklearn.utils import shuffle

import aifdb.AIFDBReader as AIFDBReader
import postagger.CitiusTagger as POSTagger
import postagger.Token as Token
import utils.PairTH as PairTH
import utils.Util as Util
from asd_en.ml import DatasetLoader
from utils.Parameters import Parameters

parameters= Parameters()

# global variable definition
stringToEntailment = {'none': 0,
                     'entailment': 1,
                     'paraphrase': 2}

entailmentToString = {v: k for k, v in stringToEntailment.items()}

class DatasetLoaderASD(DatasetLoader):


    def addLearningInstancesToDataset(self):
        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        
        dbArgMine = mongoClient.ArgMineCorpus
        
        # Sentence's table
        sentenceCollection= dbArgMine.sentence
        # delete all elements from collections before inserting new elements
        sentenceCollection.drop()
        
        
        # write articles content to input file, for PoS tagging
        newsArticlesIdsSorted= self.writeArticlesContentToPoSTaggerInputFile()
        
        # obtain corresponding article content parsed 
        # Note that the order in which the parsed articles are retrieved is implicit and can be retrieved in the ordered list "newsArticlesIdsSorted"
        
        # call POSTaggger in order to obtain sentences and token sequence from text
        
        # call POSTaggger in order to obtain sentences from text
        sentencesSequence= POSTagger.getPOSTaggerSentencesParserOutput(parameters.filenames["inputFilePoSTagger"])
        
        # call POSTagger to divide the original text into the different propositions and extracting corresponding sequence of tokens
        tokensSequence= POSTagger.getPOSTaggerOutputFromNewsFile(parameters.filenames["inputFilePoSTagger"])
        
        
        # obtain POS analyze -> dividing into the different propositions and extracting original text "fragmented"
        
        # create Proposition
        
        # add to data -> now will be a set of propositions and not a set of strings
        # Warning: in some places will be necessary to convert it to a set of strings, and depending in the case 
        # this string should be the original text or sequence of tokens lemma.
        
        i= 0
        
        # for each annotation iterate over all sentences
        for annotationSentences in sentencesSequence:
            
            # Current Article Id
            currentArticleId= newsArticlesIdsSorted[i]
            
            
            # get gold standard annotation for current article
            
            
            # obtain json structure from "currentFileName" json file
            aifJsonContent= ""
            
            with open(parameters.paths["ArgMineCorpusGoldAnnotations"] + "/" + str(currentArticleId) + "_gold" + ".json", 'r') as jsonFile:
                for line in jsonFile:
                    aifJsonContent= aifJsonContent + line.decode('utf8')
                    
            
            jsonStructure= json.loads(aifJsonContent.replace('\t', '\\t'))
            
            # obtain graph structure
            currentAIFGraph= AIFDBReader.getGraphFromJson(jsonStructure)
            
            
            j= 0
            
            # for each sentence for this specific annotation
            for sentence in annotationSentences:
                
                # add proposition to db
                currentSentence= {
                    # "sentenceId" is the absolute position of the sentence in the news article
                    "sentenceId": int(j),
                    # "articleId" is a foreign key to the article id where this sentence was extracted from
                    "articleId": int(currentArticleId),
                    # original text extracted from the article body
                    "originalText": sentence,
                    # set of tokens, where each token has the following format: (content, lemma, tags)
                    "tokens": [{"content": token.content, "lemma": token.lemma, "tags": token.tags} for token in tokensSequence[i][j]]
                    }
                
                sentenceCollection.insert_one(currentSentence)
                
                # add tuple (docId, propositionId) to dataset, data field
                ((self.dataset).data).append( (currentArticleId, j) )
                
                # add current sentence target value to dataset, target field
                
                currentSentenceIsArgument= self.determineSentenceTargetValue(sentence, currentAIFGraph)
                
                if currentSentenceIsArgument:
                    ((self.dataset).target).append(1)
                else:
                    ((self.dataset).target).append(0)
                
                j += 1
                
            
            i += 1
        
        # add target names to dataset bunch
        (self.dataset).target_names= ['no argument', 'argument']
        
        # create indexes for database queries
        #sentenceCollection.createIndex({"articleId": 1, "sentenceId": 1})
        
        # close connection to database
        mongoClient.close()
        
        # get Textual Entailment predictions for each sentence in the dataset
        #self.getTextualEntailmentPredictions((self.dataset).data)
        
    
    # output: bunch of the training set divided by percentage of articles "self.trainingSetPercentage" to include in the training set
    #TODO: [WARNING] In this version we manually force the news articles to be included in the test set
    def getTainingTestSetSplit(self, trainingSetPercentageSplit= 0.8, randomStateSeed= 12345):
        
        trainingSet= Bunch()
        testSet= Bunch()
        
        articleIdsSet= shuffle(list(set([elem[0] for elem in (self.dataset).data])), random_state= randomStateSeed)
        
        articlesIdsTestSet= [3, 95, 30, 15, 43, 49] #articleIdsSet[0:int(math.floor(( ((1.0 - trainingSetPercentageSplit) * len(articleIdsSet)) - 1)))]
        
        trainingSet.data= [elem for elem in (self.dataset).data if elem[0] not in articlesIdsTestSet]
        trainingSet.target= [(self.dataset).target[elemIndex] for elemIndex in xrange(len((self.dataset).target)) if (self.dataset).data[elemIndex][0] not in articlesIdsTestSet]
        trainingSet.target_names= (self.dataset).target_names
        
        testSet.data= [elem for elem in (self.dataset).data if elem[0] in articlesIdsTestSet]
        testSet.target= [(self.dataset).target[elemIndex] for elemIndex in xrange(len((self.dataset).target)) if (self.dataset).data[elemIndex][0] in articlesIdsTestSet]
        testSet.target_names= (self.dataset).target_names
        
        return (trainingSet, testSet)

    def determineSentenceTargetValue(self, sentence, annotation):
        currentSentenceIsArgument= False
        
        for node in annotation.nodes:
            if (node.nodeInfo).nodeType == 'I':
                #TODO: there is a build in function that is faster and may do the same thing
                if Util.relaxedSubStringSimilarity((node.nodeInfo).text, sentence, 0.85):
                    currentSentenceIsArgument= True
                    break
        
        return currentSentenceIsArgument
    
    # retrieve articles in database, sort them in a specific order and write the content ("body" element) of each article in a text file that will be given as input
    # to the PoS Tagger: in the order previously defined and separated by a special char to distinguish each article in the corresponding output file
    # Output: list of article ids in the same order in which they were written in the input file
    def writeArticlesContentToPoSTaggerInputFile(self):
        
        # connect to database
        mongoClient = MongoClient('localhost', 27017)
        
        dbArgMine = mongoClient.ArgMineCorpus
        
        # Article's table
        articleCollection= dbArgMine.article
        
        # TODO: create function to do this job
        # create file that will contain all the complete news -> each news is separated by the special symbol "@@@@@"
        # This way we can speed up the time taken by the POSTagger Tool
        # write news content into file (in order to be used by PoSTagger posteriorly) at 'taggerInputPath'
        inputFilePoSTagger = codecs.open(filename= parameters.paths["taggerInput"] + "/" + parameters.filenames["inputFilePoSTagger"] + ".txt", mode= 'w', encoding="utf-8")
        
        
        # Note: The implicit order of the argument diagrams in the "argumentDiagrams" list and corresponding news text in 
        # "newsFile" text file is assumed to be the same by the POSTagger
        articlesSortedList= articleCollection.find({}).sort([("_id", 1)])
        
        newsArticlesIdsSorted= []
        
        for article in articlesSortedList:
            
            newsArticlesIdsSorted.append(article["_id"])
            
            # writing original news text into "newsFile" file to be processed by POSTagger
            inputFilePoSTagger.write(article["body"])
            inputFilePoSTagger.write("\n" + "@@@@@" + "\n")
            
        
        inputFilePoSTagger.close()
        
        return newsArticlesIdsSorted

    # get Textual Entailment predictions for each sentence in the dataset
    def getTextualEntailmentPredictions(self, X):
        
        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        
        dbArgMine = mongoClient.ArgMineCorpus
        
        # Sentence's table
        sentenceCollection= dbArgMine.sentence
        
        # Textual Entailment's table
        textualEntailmentCollection = dbArgMine.textualEntailmentPrediction
        # delete all elements from collections before inserting new elements
        textualEntailmentCollection.drop()
        
        # Load Textual Entailment Model
        teModelLoadTimeStart= time.time()
        
        # get entailment predictions from textual entailment model
        teModel = joblib.load(parameters.paths["externalModels"] + "/" + parameters.filenames["textualEntailmentModel"])
        
        elapsedTimeTEModelLoad= time.time() - teModelLoadTimeStart
        print "\nTextual Entailment Model loaded in " + str(elapsedTimeTEModelLoad) + " sec.]\n"
        
        # Training data
        
        generateTextEntailExamplesTimeStart= time.time()
        
        # generate all combinations of T-H pairs
        allCombinationsOfTextHypothesisPairs= []
        
        # sentenceTuple= (docId, sentId)
        for sentenceTuple in X:
            #remainingSentenceIdsInArticle= [sent[1] for sent in (self.dataset).data if (sent[0] == sentenceTuple[0]) and (not (sent[1] == sentenceTuple[1]))]
            currentSentence= sentenceCollection.find_one({"$and":[{"articleId": sentenceTuple[0]}, {"sentenceId": sentenceTuple[1]}]})
            
            remainingSentenceInArticle= sentenceCollection.find({"$and":[{"articleId": sentenceTuple[0]}, {"sentenceId": {"$ne": sentenceTuple[1]}}]})
            
            
            for sent in remainingSentenceInArticle:
                
                currentKey= str(sentenceTuple[0]) + "_T" + str(currentSentence["sentenceId"]) + "_H" + str(sent["sentenceId"])
                
                
                allCombinationsOfTextHypothesisPairs.append(PairTH.PairTH(
                                                                    currentKey, 
                                                                    0, 
                                                                    currentSentence["originalText"],
                                                                    sent["originalText"],
                                                                    [Token.Token(tokenTuple["content"].encode("utf-8"), tokenTuple["lemma"].encode("utf-8"), tokenTuple["tags"]) for tokenTuple in currentSentence["tokens"]],
                                                                    [Token.Token(tokenTuple["content"].encode("utf-8"), tokenTuple["lemma"].encode("utf-8"), tokenTuple["tags"]) for tokenTuple in sent["tokens"]],
                                                                    ))
                
                currentKey= str(sentenceTuple[0]) + "_T" + str(sent["sentenceId"]) + "_H" + str(currentSentence["sentenceId"])
                
                
            
        
        elapsedTimeGenerateTextEntailExamples= time.time() - generateTextEntailExamplesTimeStart
        print "\nGenerated Textual Entailment Examples in " + str(elapsedTimeGenerateTextEntailExamples) + " sec.]\n"
        
        
        tePredictionsTimeStart= time.time()
        
        # Textual Entailment Predictions
        tePredictions= teModel.predict(allCombinationsOfTextHypothesisPairs)
        teProbaPredictions= teModel.predict_proba(allCombinationsOfTextHypothesisPairs)
        
        elapsedTimeTEPredictions= time.time() - tePredictionsTimeStart
        print "\nTextual Entailment Predictions in " + str(elapsedTimeTEPredictions) + " sec.]\n"
        
        
        # Textual Entailment Predictions -> resolution step
        teInfoAssignmentTimeStart= time.time()
        
        for sentenceTuple in (self.dataset).data:
            
            tePredictionsForCurrentSentence= []
            
            for pairTHIndex in xrange(len(allCombinationsOfTextHypothesisPairs)):
                currentIdSplitted= (allCombinationsOfTextHypothesisPairs[pairTHIndex].id).split("_")
                
                if (int(currentIdSplitted[0]) == int(sentenceTuple[0])):
                    if int(currentIdSplitted[1][1:]) == int(sentenceTuple[1]):
                        
                        tePredictionsForCurrentSentence.append({
                            "targetSentenceId": int(currentIdSplitted[2][1:]),
                            "targetSentenceRole": "hypothesis",
                            "prediction": entailmentToString[tePredictions[pairTHIndex]],
                            "probaPredictions": {"none": teProbaPredictions[pairTHIndex][0], "entailment": teProbaPredictions[pairTHIndex][1], "paraphrase": teProbaPredictions[pairTHIndex][2]}
                        })
                        
                    elif int(currentIdSplitted[2][1:]) == int(sentenceTuple[1]):
                        
                        tePredictionsForCurrentSentence.append({
                            "targetSentenceId": int(currentIdSplitted[1][1:]),
                            "targetSentenceRole": "text",
                            "prediction": entailmentToString[tePredictions[pairTHIndex]],
                            "probaPredictions": {"none": teProbaPredictions[pairTHIndex][0], "entailment": teProbaPredictions[pairTHIndex][1], "paraphrase": teProbaPredictions[pairTHIndex][2]}
                        })
                        
                    
                    
            
            # add predictions to database
            sentenceEntailmentPredictionRow= {
                    "sentenceId": int(sentenceTuple[1]),
                    "articleId": int(sentenceTuple[0]),
                    "predictions": tePredictionsForCurrentSentence
                    }
            
            textualEntailmentCollection.insert_one(sentenceEntailmentPredictionRow)
            
        
        
        elapsedTimeTEInfoAssignment= time.time() - teInfoAssignmentTimeStart
        print "\nTextual Entailment Predictions added to database in " + str(elapsedTimeTEInfoAssignment) + " sec.]\n"
        
        
        # create indexes for database queries
        #textualEntailmentCollection.createIndex({"articleId": 1, "sentenceId": 1})
        
        # close database connection
        mongoClient.close()
        
    
    
    
    
    
    def getDatasetInfo(self):
        
        numberOfSentences= 0
        numberOfArgumentativeSentences= 0
        
        for i in range(0, len((self.dataset).data)):
            
            numberOfSentences += 1
            
            if (self.dataset).target[i] == 1:
                numberOfArgumentativeSentences += 1
        
        print "no. argumentative sentences= " + str(numberOfArgumentativeSentences)
        print "no. not argumentative sentences= " + str(numberOfSentences - numberOfArgumentativeSentences)
        print "total no. of sentences= " + str(numberOfSentences)






"""
### Test ###
myDatasetLoader= DatasetLoaderASD()
#print myDatasetLoader
myDatasetLoader.getDatasetInfo()
"""

