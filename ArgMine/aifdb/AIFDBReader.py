#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AIFDBReader: set of functions to extract all the necessary information from AIFDB and 
organize it into proper data structures

Assumptions:
- "getArgumentDiagram" function when reading the text document (.txt) assumes that only one line exists
- db file in "../data/db" folder should have the following name: "argmine.db"

"""

import os
import json
import sqlite3
import codecs

import utils.Parameters as Parameters
import utils.Graph as Graph
import utils.Node as Node
import utils.Edge as Edge
import NodeInfo
import EdgeInfo
import ArgumentDiagram

# paths
parameters= Parameters.Parameters()


# Functions

def getAllFilenamesFromDir(path):
    
    filesName= []
    
    for textFile in os.listdir(path):
        if textFile.endswith(".txt"):
            currentFileName= textFile.split(".txt")[0]
            filesName.append(currentFileName)
    return filesName


# get argument diagram from "currentFileName" text file
#
# 'currentFileName.txt' was generated by AIFDB and contains the web address of the 
# original news -> from this file it is necessary to extract the 'newsid'
#
# in order to obtain the corresponding argument diagram we have to extract the information 
# contained in the json file with the same name
def getArgumentDiagram(filepath, filename):
    
    # Obtain 'newsid' from 'currentFileName.txt' text file
    # text file content is: 'http://web.fe.up.pt/~ei11124/argmine_news/newsid.html'
    
    newsId= ""
    
    with open(filepath + "/" + filename + ".txt", 'r') as fileText:
        # Note that only one line should exist 
        for line in fileText:
            splitedLine= line.split(".html")
            newsId= splitedLine[0].split("/")[-1]
    
    # obtain json structure from "currentFileName" json file
    jsonContent= ""
    
    with open(filepath + "/" + filename + ".json", 'r') as jsonFile:
        for line in jsonFile:
            jsonContent= jsonContent + line.decode('utf8')
            
    
    jsonStructure= json.loads(jsonContent.replace('\t', '\\t'))
    
    # obtain graph structure
    graph= getGraphFromJson(jsonStructure)
    
    # obtain original and complete text
    
    # known PyDev problem: has difficulties in recognizing some methods from external libraries 
    # it is necessary to add a comment indicating that we expect that error and 
    # that this error should be ignore by the interpreter
    conn= sqlite3.connect(paths["database"] + "/" + filenames["argmineDatabase"])  # @UndefinedVariable
    
    # sqlite3 offers a built-in optimized text_factory that will return bytestring
    # objects, if the data is in ASCII only, and otherwise return unicode objects
    conn.text_factory= sqlite3.OptimizedUnicode # @UndefinedVariable
    
    c= conn.cursor()
    
    c.execute('SELECT newsBody FROM News WHERE id_news = ?', (newsId,))
    
    completeText= c.fetchone()
    
    # after we obtain the first (and unique) tuple, we need to encode his content in 'utf8'
    completeText = completeText[0].encode('utf8')
    
    # create Argument Diagram and return it
    return ArgumentDiagram.ArgumentDiagram(graph, completeText, newsId, filename)


def getGraphFromJson(jsonStructure):
    
    # obtain all edges
    edgesList= []
    
    for edgeStructure in jsonStructure["edges"]:
        edgesList.append(Edge.Edge(EdgeInfo.EdgeInfo(edgeStructure["edgeID"], edgeStructure["formEdgeID"]), edgeStructure["toID"]))
    
    # obtain all Nodes
    nodesList= []
    
    for nodeStructure in jsonStructure["nodes"]:
        
        # type "L" nodes should be ignored
        # they are accessory nodes indicating the origin (web address) of the node content
        if nodeStructure["type"] != "L":
            
            nodeID= nodeStructure["nodeID"]
            
            adjEdges= []
            
            # get all edges from current node
            for edgeStructure in jsonStructure["edges"]:
                if edgeStructure["fromID"] == nodeID:
                    currentEdgeId= edgeStructure["edgeID"]
                    
                    # get edge object
                    for e in edgesList:
                        if e.getEdgeInfo().getEdgeId() == currentEdgeId:
                            adjEdges.append(e)
                            break
            
            # obtain scheme attribute (if exists)
            schemeValue= None
            if nodeStructure.get("scheme") is not None:
                schemeValue= nodeStructure["scheme"]
            
            schemeIDValue= None
            if nodeStructure.get("schemeID") is not None:
                schemeIDValue= nodeStructure["schemeID"]
            
            nodesList.append(Node.Node(NodeInfo.NodeInfo(nodeID, nodeStructure["text"].encode('utf8'), nodeStructure["type"], nodeStructure["timestamp"], schemeValue, schemeIDValue), adjEdges))
            
    
    return Graph.Graph(nodesList)
    
    
def getAllNewsFromCorpus():
    
    # obtain original and complete text
    
    
    
    # known PyDev problem: has difficulties in recognizing some methods from external libraries 
    # it is necessary to add a comment indicating that we expect that error and 
    # that this error should be ignore by the interpreter
    conn= sqlite3.connect(paths["database"] + "/argmine.db")  # @UndefinedVariable
    
    # sqlite3 offers a built-in optimized text_factory that will return bytestring
    # objects, if the data is in ASCII only, and otherwise return unicode objects
    conn.text_factory= sqlite3.OptimizedUnicode # @UndefinedVariable
    
    c= conn.cursor()
        
    c.execute('SELECT newsBody FROM News ORDER BY id_news ASC')
    
    completeTexts= c.fetchall()
    
    i = 1
    
    taggerInputFile = open(parameters.paths["taggerInput"] + "/" + "allNews" + '.txt','w')
    
    for completeText in completeTexts:
        
        # after we obtain the first (and unique) tuple, we need to encode his content in 'utf8'
        completeText = completeText[0].encode('utf8')
        
        # write news content into file (in order to be used by PoSTagger posteriorly) at 'taggerInputPath'
        
        taggerInputFile.write(completeText)
        taggerInputFile.write("\n" + "@@@@@" + "\n")
        
        
        i = i + 1
        
    taggerInputFile.close()
    

def graphToAIF(graph):
    
    aifJsonStructure= {"nodes": [], "edges": [], "locutions": []}
    
    for node in graph.nodes:
        # add node to output
        if ((node.nodeInfo).nodeType == 'I') or ((node.nodeInfo).nodeType == 'L'):
            aifJsonStructure["nodes"].append({
                "nodeID": (node.nodeInfo).nodeId,
                "text": (node.nodeInfo).text,
                "type": (node.nodeInfo).nodeType,
                "timestamp": (node.nodeInfo).timestamp,
                })
        else:
            aifJsonStructure["nodes"].append({
                "nodeID": (node.nodeInfo).nodeId,
                "text": (node.nodeInfo).text,
                "type": (node.nodeInfo).nodeType,
                "timestamp": (node.nodeInfo).timestamp,
                "scheme": (node.nodeInfo).scheme,
                "schemeID": (node.nodeInfo).schemeID
                })
        
        # add current node edges to output
        for edge in node.edges:
            aifJsonStructure["edges"].append({
                "edgeID": (edge.edgeInfo).edgeId,
                "fromID": (node.nodeInfo).nodeId,
                "toID": edge.destinationNodeId,
                "formEdgeID": (edge.edgeInfo).formEdgeId
                })
        
    
    return aifJsonStructure
    


def argumentDiagramToAIFdbFiles(argumentDiagram, destinationPath):
    
    # aif annotation file
    jsonFile = codecs.open(filename= destinationPath + "/" + str(argumentDiagram.newsId) + "_gold.json", mode= "w", encoding="utf-8")
    
    json.dump(graphToAIF(argumentDiagram.graph), jsonFile)
    
    jsonFile.close()
    
    # text file -> article url source
    textFile = codecs.open(filename= destinationPath + "/" + str(argumentDiagram.newsId) + "_gold.txt", mode= "w", encoding="utf-8")
    
    textFile.write("http://web.fe.up.pt/~argmine/argmine_news/" + str(argumentDiagram.newsId) + ".html")
    
    textFile.close()
    



"""
### Test ###
ad= getArgumentDiagram("nodeset4575")
print ad
"""
