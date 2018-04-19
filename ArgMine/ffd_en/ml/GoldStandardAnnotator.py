#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GoldStandardCorpus generator
"""


import aifdb.AIFDBReader as AIFDBReader
import utils.Parameters as Parameters
import utils.Util as Util

from pymongo import MongoClient

parameters= Parameters()
paths= parameters.paths
filenames= parameters.filenames


class GoldStandardAnnotator:
    
    def __init__(self):
        
        # dictionary where keys correspond to the ids of articles and values correspond to the annotation filename ids made for the corresponding article
        self.articleAnnotationsDict= {}
        
        # dictionary containing the gold standard annotations
        # dictionary format: key= articleId; body= {"annotation": annotation structure, "scoreIAA": float}
        self.goldStandardAnnotations= {}

    def getAnnotationsFromDir(self):
        
        # obtain all argument diagrams
        argmineCorpusFilenames= AIFDBReader.getAllFilenamesFromDir(paths["ArgMineCorpus"])
        
        # obtain all argument diagrams (note that "AIFDBReader.getArgumentDiagram" automatically creates one text file at "taggerInput" folder for each 
        # news whose content is the corresponding original text)
        for filename in argmineCorpusFilenames:
            ad= AIFDBReader.getArgumentDiagram(filename)
            
            # add current argument diagram to list of arguments
            if int(ad.newsId) in self.articleAnnotationsDict:
                self.articleAnnotationsDict[int(ad.newsId)].append(ad)
            else:
                self.articleAnnotationsDict[int(ad.newsId)] = [ad]

    def insertAnnotatedArticlesInDatabase(self):
        
        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        
        dbArgMine = mongoClient.ArgMineCorpus
        
        # Article's table
        articleCollection = dbArgMine.article
        # delete all elements from collections before inserting new elements
        articleCollection.drop()
        
        
        for newsId in (self.articleAnnotationsDict).keys():
            
            
            # add news to mongo db
            if articleCollection.find({"_id": int(self.articleAnnotationsDict[newsId][0].newsId)}).count() == 0:
                currentArticle= {
                    "_id": int(newsId),
                    "body": self.articleAnnotationsDict[newsId][0].originalText
                    }
                
                articleCollection.insert_one(currentArticle)
            
        mongoClient.close()


    # Input: "strategy" used to determine the Gold Standard annotation for each article in "self.articleAnnotationsDict"
    # Output: dictionary with the following format:
    #             - key: article identifier (same as "self.articleAnnotationsDict")
    #             - Value: Gold Standard Annotation in a dictionary format
    def getGoldStandardAnnotations(self, strategy= 1):
        
        articleAnnotationGoldStandard= {}
        if strategy == 1:
            for articleId in (self.articleAnnotationsDict).keys():
                #currenArticleIAAMetricsByAnnotation
                metricsIAA= {}
                
                for currentAnnotationIndex in range(len((self.articleAnnotationsDict[articleId]))):
                    # Argument Diagram for current annotation
                    currentArgumentDiagram= (self.articleAnnotationsDict)[articleId][currentAnnotationIndex]
                    
                    # Annotated nodes for current Argument Diagram
                    currentArgumentDiagramNodes= [node for node in currentArgumentDiagram.graph.nodes if (node.nodeInfo).nodeType == 'I']
                    
                    # list indexes for all remaining annotations
                    remainingIndexes= list(set(range(len((self.articleAnnotationsDict[articleId])))) - set([currentAnnotationIndex]))
                    
                    # current annotation IAA metrics for each remaining annotation
                    # key: other annotation filename
                    # value: IAA metrics obtained by comparing the current annotation with other annotation made for the same article
                    currentArgumentDiagramIAAMetrics= {}
                    
                    # loop over all the remaining annotations
                    for otherAnnotationIndex in remainingIndexes:
                        
                        otherArgumentDiagram= (self.articleAnnotationsDict)[articleId][otherAnnotationIndex]
                        
                        otherArgumentDiagramNodes= [node for node in otherArgumentDiagram.graph.nodes if (node.nodeInfo).nodeType == 'I']
                        
                        # number of matched nodes in the current annotation with annotated nodes in other annotation
                        numberOfMatchedNodes= 0
                        
                        for currentNode in currentArgumentDiagramNodes:
                            # current node matches any node from other annotation
                            #TODO: simplification was made here -> it should not match by only checking the content -> we should also check if the node is used in the same way
                            # (e.g. as a premise to support a specific conclusion)
                            matchedNode= False
                            
                            for nodeFromOtherArgumentDiagram in otherArgumentDiagramNodes:
                                
                                if Util.relaxedStringSimilarity((currentNode.nodeInfo).text, (nodeFromOtherArgumentDiagram.nodeInfo).text, 0.75):
                                    matchedNode= True
                                    break
                            
                            if matchedNode:
                                numberOfMatchedNodes += 1
                            
                        
                        # IAA metrics (currentAnnotationIndex -> otherAnnotationIndex)
                        
                        # Precision: number of matched nodes in "currentAnnotationIndex" from all the nodes annotated in "currentAnnotationIndex"
                        precision= 0.0
                        
                        if len(currentArgumentDiagramNodes) > 0:
                            if numberOfMatchedNodes > len(currentArgumentDiagramNodes):
                                print ("\n[WARNING] Number of Matched Nodes > Number of Nodes in Argument Diagram! This may occur when 2 nodes from an annotator are included in 1 node from the other -> similarity between annotated nodes should be analyzed\n")
                                precision= 1.0
                            else:
                                precision= float(numberOfMatchedNodes) / float(len(currentArgumentDiagramNodes))
                        
                        # Recall: number of matched nodes in "currentAnnotationIndex" from all the nodes annotated in "otherAnnotationIndex"
                        recall= 0.0
                        
                        if len(otherArgumentDiagramNodes) > 0:
                            if numberOfMatchedNodes > len(otherArgumentDiagramNodes):
                                print( "\n[WARNING] Number of Matched Nodes > Number of Nodes in Argument Diagram! This may occur when 2 nodes from an annotator are included in 1 node from the other -> similarity between annotated nodes should be analyzed\n")
                                recall= 1.0
                            else:
                                recall= float(numberOfMatchedNodes) / float(len(otherArgumentDiagramNodes))
                        
                        # F1-Score= (2 * Precision * Recall) / (Precision + Recall)
                        f1score= 0.0
                        if (precision + recall) > 0.0:
                            f1score= float(2.0 * precision * recall) / float(precision + recall)
                        
                        # add IAA metrics for "currentAnnotationIndex" annotation
                        # each IAA metric is a dictionary with the following format:
                        # key= "otherAnnotationIndex" annotation filename -> other annotation that was used to obtain the IAA metrics for "currentAnnotationIndex" annotation
                        # Value= dictionay of metrics
                        currentArgumentDiagramIAAMetrics[otherArgumentDiagram.filename]= {
                            "precision": precision,
                            "recall": recall,
                            "f1score": f1score
                            }
                    
                    metricsIAA[currentArgumentDiagram.filename] = currentArgumentDiagramIAAMetrics
                    
                
                # From all the annotations, determine the annotation with highest IAA metrics
                # Best annotation criterion: annotation with highest mean F1-Score comparing with all the annotation made for the target article. We select the annotation (one of the 
                # annotation that was made, it is not an artificial annotation) that contains more shared nodes with other annotations made for the target article
                bestMeanF1score= 0.0
                bestMeanF1scoreFilename= None
                
                
                
                for currentAnnotationFilename, iaaMetricsWithOtherAnnotations in metricsIAA.items():
                    f1scoreAccumulator= 0.0
                    counter= 0
                    for otherAnnotationFilename, currentOtherIAAMetrics in iaaMetricsWithOtherAnnotations.items():
                        f1scoreAccumulator += currentOtherIAAMetrics["f1score"]
                        counter += 1
                    
                    
                    currentMeanScore= 0.0
                    if counter > 0:
                        currentMeanScore = float(f1scoreAccumulator) / float(counter)
                    
                    if currentMeanScore >= bestMeanF1score:
                        bestMeanF1score = currentMeanScore
                        bestMeanF1scoreFilename= currentAnnotationFilename
                
                if bestMeanF1scoreFilename is not None:
                    # selected annotation for target article
                    # key: target article identifier
                    # value: dictionary containing:
                    #     - "filenameId": gold standard annotation id
                    #     - "score": IAA mean F1-Score
                    goldStandardArgumentDiagram= None
                    for ad in self.articleAnnotationsDict[articleId]:
                        if ad.filename == bestMeanF1scoreFilename:
                            goldStandardArgumentDiagram= ad
                            break
                    articleAnnotationGoldStandard[articleId] = {"annotation": goldStandardArgumentDiagram, "scoreIAA": bestMeanF1score}
                else:
                    raise Exception("Gold Standard Annotation not found @ GoldStandardAnnotator.getGoldStandardAnnotations()!")
                
            
            
            return articleAnnotationGoldStandard
        else:
            raise Exception("Invalid parameter @ GoldStandardAnnotator.getGoldStandardAnnotations()!")

    def goldStandardCorpusToAIFdbFiles(self, destinationPath):
        for articleId, goldStandardAnnotation in (self.goldStandardAnnotations).items():
            AIFDBReader.argumentDiagramToAIFdbFiles(goldStandardAnnotation["annotation"], destinationPath)
    

####################
#####   MAIN   #####
####################

gsAnnotator= GoldStandardAnnotator()

gsAnnotator.getAnnotationsFromDir()

print ("\nDictionary news-annotation:")
for x,y in (gsAnnotator.articleAnnotationsDict).items():
    print ("newsId= " + str(x) + " -> annotations= " + str([argDiagram.filename for argDiagram in y]) + " !")

gsAnnotator.insertAnnotatedArticlesInDatabase()

gsAnnotator.goldStandardAnnotations= gsAnnotator.getGoldStandardAnnotations()

print ("\nDictionary news-annotation:")
for x,y in (gsAnnotator.goldStandardAnnotations).items():
    print ("newsId= " + str(x) + " -> Gold Standard Annotation Filename= " + str(y["annotation"].filename) + " -> IAA Score= " + str(y["scoreIAA"]) + " !")

gsAnnotator.goldStandardCorpusToAIFdbFiles(paths["ArgMineCorpusGoldAnnotations"])