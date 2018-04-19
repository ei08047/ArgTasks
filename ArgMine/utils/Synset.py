#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Synset: 
"""

import copy
import os
import codecs

class Synset:
    
    def __init__(self, id, pos, word, confidenceValue, synsetsList):
        # Document ID
        self.id= id
        
        self.pos= pos
        
        self.word= word
        
        self.confidenceValue= confidenceValue
        
        self.synsetsList= synsetsList
        
        
        
    def __str__(self):
        
        stringOutput = "id= " + str(self.id) + ", pos= " + str(self.pos) + ", word= " + str(self.word.encode("utf-8")) + ", confidenceValue= " + str(self.confidenceValue) + ", synsetsList= ["
        
        for syn in self.synsetsList:
            stringOutput += "(" + str(syn[0].encode("utf-8")) + ", " + str(syn[1]) + ") / "
        
        stringOutput += "]\n"
        return stringOutput
        
    def getWordsAndConfidenceScores(self):
        
        wordsAndConfidenceScoresInSynset= [[],[]]
        
        wordsAndConfidenceScoresInSynset[0].append(self.word)
        wordsAndConfidenceScoresInSynset[1].append(float(self.confidenceValue))
        
        for word, confScore in self.synsetsList:
            wordsAndConfidenceScoresInSynset[0].append(word)
            wordsAndConfidenceScoresInSynset[1].append(float(confScore))
        
        return wordsAndConfidenceScoresInSynset
        