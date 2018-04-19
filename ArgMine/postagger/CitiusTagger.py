#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CitiusTagger: set of functions to run and extract the outputs produced by the 
POS Tagger tool (CitiusTagger)

Assumptions:
- "citiusTaggerPath" corresponds to the absolute path of the CitiusTagger installation 
directory. Because it is an external resource the path may vary depending of the installation directory

"""

import os
import subprocess
import Token
from utils.Parameters import Parameters
from pymongo import MongoClient

# paths
parameters= Parameters()

# obtain parent directory from current file absolute path
currentPath = os.path.dirname(os.path.abspath(__file__))

# Functions

def executePOSTagger(inputFileName):
    
    # change current dir to "citiusTaggerPath" dir in order to be able to run the 
    # postagger program (because it is a shell script)
    os.chdir(parameters.paths["citiusTagger"])
    
    # create output file -> which will receive the PoS tagger output 
    posTaggerOutputFile = open(parameters.paths["taggerOutput"] + "/" + inputFileName + 'Tagged.txt', 'w') 
    # execute postagger
    subprocess.call(['./nec.sh', 'pt', parameters.paths["taggerInput"] + '/' + inputFileName + '.txt'], stdout=posTaggerOutputFile)
    
    
    
    posTaggerOutputFile.close()
    
    # after execute postagger program change to current dir
    os.chdir(currentPath)




def executePOSTaggerSentencesParser(filename):
    
    # change current dir to "citiusTaggerPath" dir in order to be able to run the 
    # postagger program (because it is a shell script)
    os.chdir(parameters.paths["citiusTagger"])
    
    # create output file -> which will receive the PoS tagger output 
    posTaggerOutputFile = open(parameters.paths["taggerOutput"] + "/" + filename + "Sentences" + 'Tagged.txt', 'w') 
    # execute postagger
    subprocess.call(['./proposition.sh', 'pt', parameters.paths["taggerInput"] + "/" + filename + '.txt'], stdout=posTaggerOutputFile)
    
    posTaggerOutputFile.close()
    
    # after execute postagger program change to current dir
    os.chdir(currentPath)

# get postagger parsing (tagging) for content of 'inputFileName' text document
# Output Format: Each list corresponds to a phrase and each element corresponds to a token
# [ [Token1, Token2, ..., Tokenx]  #phrase number 1
#   [Token1, Token2, ..., Tokenx]  
#   [Token1, Token2, ..., Tokenx]] #phrase number n
def getPOSTaggerOutput(inputFileName):
    
    executePOSTagger(inputFileName)
    
    tokensList = []
    phrasesList = []
    
    # open the file in read mode and iterates its content
    with open(parameters.paths["taggerOutput"] + "/" + inputFileName + 'Tagged.txt', 'r') as fileText:
        for line in fileText:
            splitedLine = line.split()
        
            if len(splitedLine) == 0:
                # when a phrase ends, the PoSTagger outputs a newline
                
                # all tokens from current phrase have been obtained
                # add current list of tokens to phrases list
                phrasesList.append(tokensList)
                
                # empty list of tokens
                tokensList = []
                
            elif len(splitedLine) == 2:
                # when PoSTagger does not know the lemma, he only outputs the original set of words and the tags
            
                tokenContent = splitedLine[0]
                tags = splitedLine[1]
            
                currentToken = Token.Token(tokenContent, tokenContent, tags)
            
                tokensList.append(currentToken)
            else:
                # normal case -> current token have original content, lemma and tags
                tokenContent = splitedLine[0]
                lemma = splitedLine[1]
                tags = splitedLine[2]
            
                currentToken = Token.Token(tokenContent, lemma, tags)
            
                tokensList.append(currentToken)
                
    return phrasesList

# get postagger parsing (tagging) for content of 'newsFileTagged' text document
# Output Format: Each element of the list corresponds to an argument diagram. For each argument diagram exists a list where each 
# element corresponds to a proposition. For each proposition exists a list where each element corresponds to a token.
# [
#  [ [Token1, Token2, ..., Tokenx],  #proposition number 1
#    [Token1, Token2, ..., Tokenx],  
#    [Token1, Token2, ..., Tokenx]] #proposition number n
#  ], # argument diagram number 1
#  [ ...
#  ]
# ]
def getPOSTaggerOutputFromNewsFile(filename):
    
    executePOSTagger(filename)
    
    # set of tokens (original word, lemma, tags) from some sentence
    tokensList = [] 
    
    # set of sentences from some news
    annotationList = []
    
    # set of news
    annotationsList = []
    
    # auxiliary variable to parse POSTagger output
    currentState= 0
    
    
    # open the file in read mode and iterates its content
    with open(parameters.paths["taggerOutput"] + "/" + filename + 'Tagged.txt', 'r') as fileText:
        for line in fileText:
            splitedLine = line.split()
        
            if len(splitedLine) == 0:
                # when a phrase ends, the PoSTagger outputs a newline
                # this means that we already have all the tokens for the sentence 'i'  from the news 'j'
                # So, we add this set of tokens to the 'annotationList' and we go to the next sentence
                
                if len(tokensList):
                
                    # all tokens from current phrase have been obtained
                    # add current list of tokens to phrases list
                    annotationList.append(tokensList)
                
                    # empty list of tokens
                    tokensList = []
                
            elif splitedLine[0] == "@@@@@":
                # This symbol means "end of news"
                # this means that we already have all the tokens from all the sentences of the current news.
                # So, we add the set of sentences 'annotationList' to the set of news 'annotationsList'
                annotationsList.append(annotationList)
                annotationList = []
                
                # update auxiliary variable
                currentState= 1
                
            elif len(splitedLine) == 2:
                # when PoSTagger does not know the lemma, he only outputs the original set of words and the tags
                
                tokenContent = splitedLine[0]
                tags = splitedLine[1]
                
                currentToken = Token.Token(tokenContent, tokenContent, tags)
                
                tokensList.append(currentToken)
                
                
            elif (currentState == 1) and (splitedLine[0] == "."):
                # Since the POSTagger outputs a dot after the special symbol "@@@@@" we have to ignore this line of the file
                # we reset the auxiliary variable and we move to next news
                currentState= 0
                
            else:
                # normal case -> current token have original content, lemma and tags
                tokenContent = splitedLine[0]
                lemma = splitedLine[1]
                tags = splitedLine[2]
            
                currentToken = Token.Token(tokenContent, lemma, tags)
            
                tokensList.append(currentToken)
                
    return annotationsList


# obtain all sentences from corresponding argument diagram for each annotation
def getPOSTaggerSentencesParserOutput(filename):
    
    executePOSTaggerSentencesParser(filename)
    
    annotationList = []
    annotationsList = []
    
    
    # open the file in read mode and iterates its content
    with open(parameters.paths["taggerOutput"] + "/" + filename + "Sentences" + 'Tagged.txt', 'r') as fileText:
        # assuming that each line corresponds to one sentence
        for line in fileText:
            
            if line == "@@@@@\n":
                annotationsList.append(annotationList)
                annotationList = []
                
            else:
                
                annotationList.append(line)
                
                
    return annotationsList


    

def myTokenizer(proposition):
    
    lemmasList= []
    
    
    stopWordsFile= open(parameters.paths["stopWords"] + '/' + parameters.filenames["stopWords"],'r')
        
    stopWordsList= []
    
    for word in stopWordsFile:
        stopWordsList.append(word.rstrip("\n").decode("utf-8"))
    
    for token in proposition["tokens"]:
        if not ( (token["tags"][0] == 'F') or (token["lemma"] in stopWordsList ) ):
            lemmasList.append(((token["lemma"]).lower()))
    
    
    return lemmasList

def myPreProcessor(learningInstance):
    
    # connect to db
    mongoClient = MongoClient('localhost', 27017)
    
    dbArgMine = mongoClient.ArgMineCorpus
    
    # Sentence's table
    sentenceCollection= dbArgMine.sentence
    
    # return content from database
    return sentenceCollection.find_one({"$and":[{"articleId": learningInstance[0]}, {"sentenceId": learningInstance[1]}]})

