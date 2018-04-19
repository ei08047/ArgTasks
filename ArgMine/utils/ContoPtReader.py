#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ContoPt Loader
"""

import os
import time
import numpy as np

from utils.Parameters import Parameters

parameters= Parameters()
paths= parameters.paths
filenames= parameters.filenames

import Synset
import Triple

from xml.etree import cElementTree as ET


class ContoPtLoader:
    
    def __init__(self):
        
        
        # Dictionary of synsets with the following configuration (as it occurs in the original files)
        # key: synset id
        # value: Synset object
        self.synsetsMap= {}
        
        # open the file in read mode and iterates its content
        with open(paths["wordnetPT"] + "/" + filenames["synsetsWordnetPT"], 'r') as synsetsTextFile:
            
            for line in synsetsTextFile:
                
                
                splitedLine = line.split(" : ")
                
                splitedSynsets= splitedLine[2].split(";")
                
                synsetsPairs= []
                
                for s in splitedSynsets[:-1]:
                    
                    firstSplit= s.split("(")
                    
                    synsetsPairs.append((firstSplit[0].decode("utf-8"), firstSplit[1][:-1]))
                    
                
                self.synsetsMap[int(splitedLine[0])] = Synset.Synset(int(splitedLine[0]), splitedLine[1].decode("utf-8"), synsetsPairs[0][0], synsetsPairs[0][1], synsetsPairs[1:])
                
            
        
        # list of triples
        self.triples= []
        
        with open(paths["wordnetPT"] + "/" + filenames["relationsWordnetPT"], 'r') as relationsTextFile:
            
            for line in relationsTextFile:
                
                splitedLine = line.split(" ")
                
                (self.triples).append(Triple.Triple(splitedLine[0], splitedLine[2], splitedLine[1], splitedLine[4][0:-1]))
                
            
        # Dictionary with the following configuration:
        # key: unique word
        # value: set of synsets identifiers (integer) in which the key occurs
        self.wordsDict= {}
        
        self.getWordsDictFromSynsetMap()
        
        # Dictionary with the following configuration:
        # key: unique pair of synsets with the following structure: synset1Id:synset2Id
        # value: set or relations (string) that exist for the pair (synset1 relation synset2)
        self.triplesDict= {}
        
        self.getTriplesDictFromTriples()
        
        self.normalization()
        
        self.synonymsMap= self.readCandidateWordsXMLFile("synonyms")
        
        self.antonymsMap= self.readCandidateWordsXMLFile("antonyms")
        
        self.hypernymsMap= self.readCandidateWordsXMLFile("hypernyms")
        
        
        
    def __str__(self):
        
        stringOutput= ""
        
        for key, value in (self.synsetsMap).iteritems():
            
            stringOutput += "id= " + str(key) + " / Content= " + str(value) + "\n"
            
        
        return stringOutput
        
    
    # updates the dictionary self.wordDict with the same synsets included in self.synsetsMap but
    # with a different structure:
    # keys: word
    # values: [[synsetId1, synsetId2, ..., synsetIdn], [conf(word, synsetId1), conf(word, synsetId2), ..., conf(word, synsetIdn)]] 
    def getWordsDictFromSynsetMap(self):
        
        for key, value in (self.synsetsMap).iteritems():
            
            if value.word in self.wordsDict:
                (self.wordsDict[value.word])[0].append(int(key))
                (self.wordsDict[value.word])[1].append(float(value.confidenceValue))
            else:
                self.wordsDict[value.word] = [[int(key)], [float(value.confidenceValue)]]
            
            
            for e in value.synsetsList:
                currentWord= e[0]
                if currentWord in self.wordsDict:
                    (self.wordsDict[currentWord])[0].append(int(key))
                    (self.wordsDict[currentWord])[1].append(float(e[1]))
                else:
                    self.wordsDict[currentWord] = [[int(key)], [float(e[1])]]
                
            
        
    
    def getTriplesDictFromTriples(self):
        
        for triple in self.triples:
            
            currentKey = str(triple.id1) + ":" + str(triple.id2)
            
            if currentKey in self.triplesDict:
                (self.triplesDict[currentKey])[0].append(triple.relationType)
                (self.triplesDict[currentKey])[1].append(float(triple.confidenceValue))
            else:
                self.triplesDict[currentKey] = [[triple.relationType], [float(triple.confidenceValue)]]
            
        
    
    def normalization(self):
        
        # wordsDict
        maxConfidenceScore= -1
        for k, v in (self.wordsDict).iteritems():
            
            for valueIndex in xrange(len(v[1])):
                currentScore= float(v[1][valueIndex])
                if currentScore > maxConfidenceScore:
                    maxConfidenceScore = currentScore
             
        
        for k, v in (self.wordsDict).items():
            for valueIndex in xrange(len(v[1])):
                v[1][valueIndex]= v[1][valueIndex] / float(maxConfidenceScore)
            
        
        for k, v in (self.synsetsMap).items():
            
            v.confidenceValue= float(v.confidenceValue) / float(maxConfidenceScore)
            
            for wordIndex in xrange(len(v.synsetsList)):
                newTuple= (v.synsetsList[wordIndex][0], float(v.synsetsList[wordIndex][1]) / float(maxConfidenceScore))
                v.synsetsList[wordIndex]= newTuple
            
        
        
        # triplesDict
        maxConfidenceScore= -1
        for k, v in (self.triplesDict).iteritems():
            
            for valueIndex in xrange(len(v[1])):
                currentScore= float(v[1][valueIndex])
                if currentScore > maxConfidenceScore:
                    maxConfidenceScore = currentScore
             
        
        for k, v in (self.triplesDict).items():
            for valueIndex in xrange(len(v[1])):
                v[1][valueIndex]= v[1][valueIndex] / float(maxConfidenceScore)
        
    
    
    def isSynonyn(self, word1, word2):
        
        word1= word1 #.decode('ISO-8859-1')
        word2= word2 #.decode('ISO-8859-1')
        
        if word1 == word2:
            return True
        else:
            idWord1= None
            if word1 in self.wordsDict:
                idWord1= self.wordsDict[word1][0]
            
            idWord2= None
            if word2 in self.wordsDict:
                idWord2= self.wordsDict[word2][0]
            
            if (idWord1 is not None) and (idWord2 is not None):
                
                if len(set(idWord1).intersection(set(idWord2))) > 0:
                    return True
                else:
                    return False
                
            else:
                return False
        
    
    def isHyperonym(self, word1, word2):
        
        word1= word1 #.decode('ISO-8859-1')
        word2= word2 #.decode('ISO-8859-1')
        
        if word1 == word2:
            return False
        else:
            
            idWord1= None
            if word1 in self.wordsDict:
                idWord1= self.wordsDict[word1][0]
            
            idWord2= None
            if word2 in self.wordsDict:
                idWord2= self.wordsDict[word2][0]
            
            if (idWord1 is not None) and (idWord2 is not None):
                
                for id1 in idWord1:
                    for id2 in idWord2:
                        
                        targetKey= str(id1) + ":" + str(id2)
                        
                        if targetKey in self.triplesDict:
                            
                            tripleRelations= (self.triplesDict)[targetKey][0]
                            
                            for rel in tripleRelations:
                                if rel == "HIPERONIMO_DE":
                                    return True
                            
                        
                        
                return False
                
            else:
                return False
        
    
    def isHyponym(self, word1, word2):
        
        word1= word1 #.decode('ISO-8859-1')
        word2= word2 #.decode('ISO-8859-1')
        
        if word1 == word2:
            return False
        else:
            
            idWord1= None
            if word1 in self.wordsDict:
                idWord1= self.wordsDict[word1][0]
            
            idWord2= None
            if word2 in self.wordsDict:
                idWord2= self.wordsDict[word2][0]
            
            if (idWord1 is not None) and (idWord2 is not None):
                
                for id1 in idWord1:
                    for id2 in idWord2:
                        
                        targetKey= str(id1) + ":" + str(id2)
                        
                        if targetKey in self.triplesDict:
                            
                            tripleRelations= (self.triplesDict)[targetKey][0]
                            
                            for rel in tripleRelations:
                                if (rel == "MEMBRO_DE") or (rel == "PARTE_DE"):
                                    return True
                            
                        
                        
                return False
                
            else:
                return False
        
    
    
    def isAntonym(self, word1, word2):
        
        
        word1= word1 #.decode('ISO-8859-1')
        word2= word2 #.decode('ISO-8859-1')
        
        if word1 == word2:
            return False
        else:
            
            idWord1= None
            if word1 in self.wordsDict:
                idWord1= self.wordsDict[word1][0]
            
            idWord2= None
            if word2 in self.wordsDict:
                idWord2= self.wordsDict[word2][0]
            
            if (idWord1 is not None) and (idWord2 is not None):
                
                for id1 in idWord1:
                    for id2 in idWord2:
                        
                        targetKey= str(id1) + ":" + str(id2)
                        
                        if targetKey in self.triplesDict:
                            
                            tripleRelations= (self.triplesDict)[targetKey]
                            
                            antonymRelationsIndexes= [i for i in xrange(len(tripleRelations[0])) if (tripleRelations[0][i] == "ANTONIMO_V_DE") or (tripleRelations[0][i] == "ANTONIMO_N_DE") or (tripleRelations[0][i] == "ANTONIMO_ADJ_DE") ]
                            
                            maxConfidenceRelation= -1
                            maxConfidenceRelationValue= -1
                            
                            for antonymRelationIndex in antonymRelationsIndexes:
                                
                                currentConfidence= tripleRelations[1][antonymRelationIndex]
                                if currentConfidence > maxConfidenceRelationValue:
                                    maxConfidenceRelation= antonymRelationIndex
                                    maxConfidenceRelationValue= currentConfidence
                                    
                                
                            
                            if maxConfidenceRelationValue > 0.3:
                                return True
                            
                        
                        
                return False
                
            else:
                return False
        
    
    
    def getAntonym(self, word):
        
        if word in self.wordsDict:
            
            
            wordSynsets= self.wordsDict[word]
            
            maxConfidenceValue= -1
            candidateAntonym= ""
            maxConfidenceCandidateAntonym= -1
            candidateWords= [[],[]]
            
            targetKey= -1
            
            antonymTriples= [(k,v) for (k,v) in (self.triplesDict).iteritems() if ("ANTONIMO_V_DE" in v[0]) or ("ANTONIMO_N_DE" in v[0]) or ("ANTONIMO_ADJ_DE" in v[0])]
            
            for key, value in antonymTriples:
                
                # Retrieve target key for relations with highest confidence
                
                splitedKey= key.split(":")
                
                
                # From all synsets that the target word belongs to this is the synset that we are working with in the current relation
                wordSynset= -1
                
                # Synset that has an antonym relation with the "wordSynset" 
                targetSynset= -1
                
                if int(splitedKey[0]) in wordSynsets[0]:
                    targetSynset= int(splitedKey[1])
                    wordSynset= int(splitedKey[0])
                """
                elif int(splitedKey[1]) in wordSynsets[0]:
                    targetSynset= int(splitedKey[0])
                    wordSynset= int(splitedKey[1])
                """
                
                if targetSynset >= 0:
                    # This "triple" contains relations between synsets where one of them is the target synset ("targetSynset")
                    # For all relations of antonomy between this two synsets, 
                    # retrieve the highest confidence value for relations of this type
                    
                    confidenceValue= -1
                    
                    for relIndex in xrange(len(value[0])):
                        if value[0][relIndex] == "ANTONIMO_V_DE":
                            if value[1][relIndex] > confidenceValue:
                                confidenceValue= value[1][relIndex]
                        elif value[0][relIndex] == "ANTONIMO_N_DE":
                            if value[1][relIndex] > confidenceValue:
                                confidenceValue= value[1][relIndex]
                        elif value[0][relIndex] == "ANTONIMO_ADJ_DE":
                            if value[1][relIndex] > confidenceValue:
                                confidenceValue= value[1][relIndex]
                    
                    
                    # confidence of target word belonging to the synset "wordSynset"
                    wordSynsetConfidence= -1
                    
                    for i in xrange(len(wordSynsets[0])):
                        if int(wordSynsets[0][i]) == wordSynset:
                            wordSynsetConfidence= wordSynsets[1][i]
                            break
                    
                    # overall confidence is the average between "confidence that target word belongs to currenr synset 'wordSynset'" and 
                    # "confidence in that 'wordSynset' is related with 'targetSynset' with the target relation"
                    currentConfidence= float(wordSynsetConfidence) * float(confidenceValue) #float(wordSynsetConfidence + confidenceValue) / 2.0
                    
                    if currentConfidence > maxConfidenceValue:
                        maxConfidenceValue= currentConfidence
                        targetKey= targetSynset
                    
                    #print "maxConfidenceValue= " + str(maxConfidenceValue)
                    #print "targetKey= " + str(targetKey)
                    
            # Retrieve all the words in the synset 'targetKey'
            candidateAntonymWords= [[],[]]
            
            for antonymKey, antonymValue in (self.wordsDict).iteritems():
                
                if targetKey in antonymValue[0]:
                    antonymConfidenceValue= -1
                    for antonymSynsetsIndex in xrange(len(antonymValue[0])):
                        if antonymValue[0][antonymSynsetsIndex] == targetKey:
                            antonymConfidenceValue= antonymValue[1][antonymSynsetsIndex]
                            break
                    candidateAntonymWords[0].append(antonymKey)
                    candidateAntonymWords[1].append(antonymConfidenceValue)
                
            #candidateWords= candidateAntonymWords
            
            
            
            # return the antonyms order by weighted confidence
            if len(candidateAntonymWords[0]) > 0:
                """
                for antonymIndex in xrange(len(candidateAntonymWords[0])):
                    if candidateAntonymWords[1][antonymIndex] > maxConfidenceCandidateAntonym:
                        maxConfidenceCandidateAntonym= candidateAntonymWords[1][antonymIndex]
                        candidateAntonym= candidateAntonymWords[0][antonymIndex]
                
                return candidateAntonym
                """
                
                candidateAntonymWordsTransposedMatrix= (np.asarray(candidateAntonymWords)).T
                
                # WARNING: confidence values are strings instead of floats
                candidateAntonymWordsSorted= ((candidateAntonymWordsTransposedMatrix[candidateAntonymWordsTransposedMatrix[:,1].argsort()[::-1]]).T).tolist()
                
                return candidateAntonymWordsSorted
                
            else:
                return None
            
        
        return None
        
    
    def getCandidateWords(self, word, relations):
        relations= set(relations)
        
        if word in self.wordsDict:
            
            wordSynsets= self.wordsDict[word]
            
            maxConfidenceValue= -1
            
            targetKey= -1
            
            antonymTriples= [(k,v) for (k,v) in (self.triplesDict).iteritems() if len(relations.intersection(set(v[0]))) > 0]
            
            candidateSynsets= [[],[]]
            
            for key, value in antonymTriples:
                
                # Retrieve target key for relations with highest confidence
                
                splitedKey= key.split(":")
                
                
                # From all synsets that the target word belongs to this is the synset that we are working with in the current relation
                wordSynset= -1
                
                # Synset that has an antonym relation with the "wordSynset" 
                targetSynset= -1
                
                
                if int(splitedKey[1]) in wordSynsets[0]:
                    targetSynset= int(splitedKey[0])
                    wordSynset= int(splitedKey[1])
                
                if targetSynset >= 0:
                    # This "triple" contains relations between synsets where one of them is the target synset ("targetSynset")
                    # For all relations of antonomy between this two synsets, 
                    # retrieve the highest confidence value for relations of this type
                    
                    confidenceValue= -1
                    
                    for relIndex in xrange(len(value[0])):
                        if value[0][relIndex] in relations:
                            if value[1][relIndex] > confidenceValue:
                                confidenceValue= value[1][relIndex]
                    
                    
                    # confidence of target word belonging to the synset "wordSynset"
                    wordSynsetConfidence= -1
                    
                    for i in xrange(len(wordSynsets[0])):
                        if int(wordSynsets[0][i]) == wordSynset:
                            wordSynsetConfidence= wordSynsets[1][i]
                            break
                    
                    # overall confidence is the average between "confidence that target word belongs to currenr synset 'wordSynset'" and 
                    # "confidence in that 'wordSynset' is related with 'targetSynset' with the target relation"
                    currentConfidence= float(wordSynsetConfidence) * float(confidenceValue) #float(wordSynsetConfidence + confidenceValue) / 2.0
                    
                    candidateSynsets[0].append(targetSynset)
                    candidateSynsets[1].append(currentConfidence)
                    
                    
            
            
            # return the antonyms order by weighted confidence
            if len(candidateSynsets[0]) > 0:
                
                
                candidateSynsetsTransposedMatrix= (np.asarray(candidateSynsets)).T
                
                # WARNING: confidence values are strings instead of floats
                candidateAntonymWordsSorted= ((candidateSynsetsTransposedMatrix[candidateSynsetsTransposedMatrix[:,1].argsort()[::-1]]).T).tolist()
                
                return candidateAntonymWordsSorted
                
            else:
                return None
            
        
        return None
        
    
    
    def getSynonyms(self, word):
        
        if word in self.wordsDict:
            
            wordSynsets= self.wordsDict[word]
            
            maxConfidenceValue= -1
            
            # Retrieve all the words in the synset 'targetKey'
            candidateWords= [[],[]]
            
            for synonymKey, synonymValue in (self.wordsDict).iteritems():
                
                synsetIntersection= list(set(wordSynsets[0]).intersection(set(synonymValue[0])))
                
                if len(synsetIntersection) > 0:
                    
                    highestConfidenceScore= -1
                    
                    for synsetIndex in xrange(len(synsetIntersection)):
                        
                        for wordSynsetIndex in xrange(len(wordSynsets[0])):
                            if wordSynsets[0][wordSynsetIndex] == synsetIntersection[synsetIndex]:
                                wordConfidenceScoreOnCurrentTargetSynset= wordSynsets[1][wordSynsetIndex]
                                break
                        
                        #float(synonymValue[1][synsetIndex] + wordConfidenceScoreOnCurrentTargetSynset) / 2.0
                        currentScore= float(synonymValue[1][synsetIndex]) * float(wordConfidenceScoreOnCurrentTargetSynset)
                        
                        if currentScore > highestConfidenceScore:
                            highestConfidenceScore= currentScore
                        
                    
                    candidateWords[0].append(synonymKey)
                    candidateWords[1].append(highestConfidenceScore)
                
            
            
            # return the antonyms order by weighted confidence
            if len(candidateWords[0]) > 0:
                
                
                candidateWordsTransposedMatrix= (np.asarray(candidateWords)).T
                
                # WARNING: confidence values are strings instead of floats
                candidateWordsSorted= ((candidateWordsTransposedMatrix[candidateWordsTransposedMatrix[:,1].argsort()[::-1]]).T).tolist()
                
                return candidateWordsSorted
                
            else:
                return None
            
        
        return None
        
    
    def candidateWordsToXMLFile(self, title, relations):
        
        root = ET.Element("root")
        
        counter= 0
        
        for k, v in self.wordsDict.iteritems():
            
            print "[" + title + "]: " + str(counter)
            counter = counter + 1
            
            if counter == 3:
                break
            
            token = ET.SubElement(root, "target", name= k)
            
            candidateWords= self.getCandidateWords(k, relations)
            
            if candidateWords is not None:
                numberOfTopWordsFound= 10
                if len(candidateWords[0]) < numberOfTopWordsFound:
                    numberOfTopWordsFound= len(candidateWords[0])
                
                for i in xrange(numberOfTopWordsFound):
                    
                    ET.SubElement(token, "synset", confidence= str(candidateWords[1][i])).text = str(int(candidateWords[0][i]))
            
            
        
        tree = ET.ElementTree(root)
        tree.write(str(title) + ".xml", encoding='utf-8', xml_declaration=True)
        
    
    
    def candidateSynonymWordsToXMLFile(self):
        
        root = ET.Element("root")
        
        counter= 0
        
        for k, v in self.wordsDict.iteritems():
            
            print "[" + "synonymWords" + "]: " + str(counter)
            counter = counter + 1
            
            token = ET.SubElement(root, "target", name= k)
            
            candidateWords= self.getSynonyms(k)
            
            if candidateWords is not None:
                numberOfTopWordsFound= 10
                if len(candidateWords[0]) < numberOfTopWordsFound:
                    numberOfTopWordsFound= len(candidateWords[0])
                
                for i in xrange(numberOfTopWordsFound):
                    
                    ET.SubElement(token, "token", confidence= str(candidateWords[1][i])).text = candidateWords[0][i]
            
            
        
        tree = ET.ElementTree(root)
        tree.write("synonymWords.xml", encoding='utf-8', xml_declaration=True)
        
    
    def readCandidateWordsXMLFile(self, relation):
        
        print "\n\n--- Loading " + str(relation) + ".xml file ---\n"
        
        startTimeLoader= time.time()
        
        filename= paths["wordnetPT"] + "/" + relation + ".xml"
        
        synonymsMap= {}
        
        tree = ET.parse(filename)
        root = tree.getroot()
        
        for targetWord in root.iter('target'):
            
            targetWordAttribs = dict(targetWord.items())
            
            currentTargetWord= targetWordAttribs['name'] #.decode("utf-8")
            
            targetWordSynonymList= []
            
            if relation == "synonyms":
                for t in targetWord.iter('token'):
                    
                    currentToken= None
                    
                    currentToken= (t.text) #.decode("utf-8")
                    
                    tAttribs = dict(t.items())
                    
                    currentConfidence= float(tAttribs['confidence'])
                    
                    targetWordSynonymList.append((currentToken, currentConfidence))
                    
            else:
                for t in targetWord.iter('synset'):
                    
                    currentToken= None
                    
                    currentToken= int(t.text)
                    
                    tAttribs = dict(t.items())
                    
                    currentConfidence= float(tAttribs['confidence'])
                    
                    targetWordSynonymList.append((currentToken, currentConfidence))
                    
            
            synonymsMap[currentTargetWord] = targetWordSynonymList
            
        
        elapsedTimeLoader= time.time() - startTimeLoader
        print "\n[" + relation + ".xml file loaded in " + str(elapsedTimeLoader) + " sec.]\n"
        
        if len(synonymsMap) == 0:
            return None
        else:
            return synonymsMap
        
    
    def getTopKCandidateWords(self, relatedSynsets, k):
        
        # relatedSynsets: list of pairs of the following format: (synsetId, ConfidenceScore). Assumed to be obtained from: relatedSynsetsMap[relatedSynsets
        # k: number of words to be returned (if number of words returned (n) is less than k it means that n is the maximum number of related words found in the synset
        
        output= []
        
        
        targetSynsets= []
        for relatedSynsetsIndex in xrange(len(relatedSynsets)):
            
            currentSynset= self.synsetsMap[relatedSynsets[relatedSynsetsIndex][0]]
            
            targetSynsets.append((relatedSynsets[relatedSynsetsIndex][0], currentSynset, relatedSynsets[relatedSynsetsIndex][1], float(float(currentSynset.confidenceValue) * float(relatedSynsets[relatedSynsetsIndex][1]))))
            
        
        targetSynsets= sorted(targetSynsets, key=lambda x: x[3], reverse=True)
        
        
        for elemIndex in xrange(len(targetSynsets)):
            
            kWordsRetrieved= False
            
            currentConfidenceScore= targetSynsets[elemIndex][3]
            
            output.append((targetSynsets[elemIndex][1].word, currentConfidenceScore))
            
            if len(output) >= k:
                break
            
            for tokenContent, tokenConfidence in (targetSynsets[elemIndex][1]).synsetsList:
                
                currentConfidenceScore= targetSynsets[elemIndex][2] * float(tokenConfidence)
                
                if elemIndex < (len(targetSynsets) - 1):
                    
                    if currentConfidenceScore < targetSynsets[elemIndex + 1][3]:
                        break
                    
                
                output.append((tokenContent, currentConfidenceScore))
                
                if len(output) >= k:
                    kWordsRetrieved= True
                    break
            
            if kWordsRetrieved:
                break
            
        
        if len(output) == 0:
            return None
        else:
            return output
        
        
    
