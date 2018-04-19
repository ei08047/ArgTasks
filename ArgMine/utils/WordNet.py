#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WordNet Interface
"""

import os
import time
import numpy as np

import nltk
from nltk.corpus import wordnet as wordnet


class WordNetInterface:
    
    
    def getWordNet(self, word, postag= None):
        
        #word= word.decode("utf-8")
        
        if postag is not None:
            wordSynsets= wordnet.synsets(word, pos= treebankToWordnetPOSTags(postag))
            
            if len(wordSynsets) > 0:
                return wordSynsets[0]
            else:
                return None
        else:
            
            wordSynsets= wordnet.synsets(word)
            
            
            if len(wordSynsets) > 0:
                return wordSynsets[0]
            else:
                return None
        
    
    
    def isSynonyn(self, word1, word2):
        
        #word1= word1.decode("utf-8")
        #word2= word2.decode("utf-8")
        
        if word1 == word2:
            return True
        else:
            
            word1Synset= self.getWordNet(word1)
            word2Synset= self.getWordNet(word2)
            
            if (word1Synset is None) or (word2Synset is None):
                return False
            else:
                
                word1Lemmas= word1Synset.lemma_names()
                word2Lemmas= word2Synset.lemma_names()
                
                if len(set(word1Lemmas).intersection(set(word2Lemmas))) > 0:
                    return True
                else:
                    return False
                
        
        
    
    
    """
    def isSynonyn(self, word1, word2):
        
        if word1 == word2:
            return True
        else:
            
            word2Synset= self.getWordNet(word2)
            
            if (word2Synset is None):
                return False
            else:
                
                word2Lemmas= word2Synset.lemma_names()
                
                if word1 in word2Lemmas:
                    return True
                else:
                    return False
                
        
        
    """
    
    
    def isHypernym(self, word1, word2):
        
        #word1= word1.decode("utf-8")
        #word2= word2.decode("utf-8")
        
        
        if word1 == word2:
            return False
        else:
            
            #word1Synset= self.getWordNet(word1)
            word2Synset= self.getWordNet(word2)
            
            if (word2Synset is None):
                return False
            else:
                
                word2HypernymPaths= word2Synset.hypernym_paths()
                
                word2HypernymSynsets = []
                
                for hypernymPath in word2HypernymPaths:
                    word2HypernymSynsets= word2HypernymSynsets + hypernymPath[0:-1]
                
                word2HypernymSynsets= set(word2HypernymSynsets)
                
                word2HypernymLemmas= []
                for s in word2HypernymSynsets:
                    for lemmaNames in s.lemma_names():
                        word2HypernymLemmas.append(lemmaNames)
                
                
                if word1 in word2HypernymLemmas:
                    return True
                else:
                    return False
                
        
        
        
    
    
    def isHyponym(self, word1, word2):
        
        #word1= word1.decode("utf-8")
        #word2= word2.decode("utf-8")
        
        if word1 == word2:
            return False
        else:
            
            #word1Synset= self.getWordNet(word1)
            word2Synset= self.getWordNet(word2)
            
            
            if (word2Synset is None):
                return False
            else:
                
                word2Hyponyms= word2Synset.hyponyms()
                
                word2HyponymLemmas= []
                
                for hyponymSynset in word2Hyponyms:
                    
                    for lemma in hyponymSynset.lemma_names():
                        word2HyponymLemmas.append(lemma)
                    
                
                if word1 in word2HyponymLemmas:
                    return True
                else:
                    return False
                
        
    
    # Meronym - denotes a part of something
    def isMeronym(self, word1, word2):
        
        #word1= word1.decode("utf-8")
        #word2= word2.decode("utf-8")
        
        if word1 == word2:
            return False
        else:
            
            #word1Synset= self.getWordNet(word1)
            word2Synset= self.getWordNet(word2)
            
            
            if (word2Synset is None):
                return False
            else:
                
                word2Meronyms= word2Synset.part_meronyms() + word2Synset.substance_meronyms()
                
                word2MeronymLemmas= []
                
                for meronymSynset in word2Meronyms:
                    
                    for lemma in meronymSynset.lemma_names():
                        word2MeronymLemmas.append(lemma)
                    
                
                if word1 in word2MeronymLemmas:
                    return True
                else:
                    return False
                
        
    
    
    # Holonym - denotes a membership to something
    def isHolonym(self, word1, word2):
        
        #word1= word1.decode("utf-8")
        #word2= word2.decode("utf-8")
        
        if word1 == word2:
            return False
        else:
            
            #word1Synset= self.getWordNet(word1)
            word2Synset= self.getWordNet(word2)
            
            
            if (word2Synset is None):
                return False
            else:
                
                word2Holonyms= word2Synset.part_holonyms() + word2Synset.substance_holonyms()
                
                word2HolonymLemmas= []
                
                for holonymSynset in word2Holonyms:
                    
                    for lemma in holonymSynset.lemma_names():
                        word2HolonymLemmas.append(lemma)
                    
                
                if word1 in word2HolonymLemmas:
                    return True
                else:
                    return False
                
        
    
    # Antonym: just applies to adjectives (a) or satellite-adjectives (s) at the lemma level (instead of synset level)
    def isAntonym(self, word1, word2):
        
        #word1= word1.decode("utf-8")
        #word2= word2.decode("utf-8")
        
        if word1 == word2:
            return False
        else:
            
            #word1Synset= self.getWordNet(word1)
            word2AllSynsets= wordnet.synsets(word2)
            
            
            if len(word2AllSynsets) == 0:
                return False
            else:
                
                
                word2Antonyms= set()
                
                for synset in word2AllSynsets:
                    # If synset is adj or satelite-adj.
                    if synset.pos() in ['a', 's']:
                        # Iterating through lemmas for each synset
                        for synsetLemma in synset.lemmas():
                            # get antonyms for corresponding lemma
                            for antonym in synsetLemma.antonyms():
                                word2Antonyms.add(antonym.name())
                                
                        
                        
                
                
                if word1 in word2Antonyms:
                    return True
                else:
                    return False
                
        
    
    
    
    


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



############################################################
#############               Test               #############
############################################################
"""

wn= WordNetInterface()

word1String= "automobile"
word2String= "car"

#word1= wn.getWordNet(word1String)
#word2= wn.getWordNet(word2String)

print "\nSynonym test:\n"

synonymStartTime= time.time()

print str(word1String) + " is synonym of " + str(word2String) + "?"
if wn.isSynonyn(word1String, word2String):
    print "Yes"
else:
    print "No"


synonymElapsedTime= time.time() - synonymStartTime
print "\Synonyms retrived in " + str(synonymElapsedTime) + " sec.]\n"

word1String= "human"
word2String= "dog"

print "\n\nHypernym test:\n"

print str(word1String) + " is hypernym of " + str(word2String) + "?"
if wn.isHypernym(word1String, word2String):
    print "Yes"
else:
    print "No"

print "\n\nHyponym test:\n"

word1String= "car"
word2String= "vehicle"

print str(word1String) + " is hyponym of " + str(word2String) + "?"
if wn.isHyponym(word1String, word2String):
    print "Yes"
else:
    print "No"

print "\n\nMeronym test:\n"

word1String= "limb"
word2String= "tree"

print str(word1String) + " is Meronym of " + str(word2String) + "?"
if wn.isMeronym(word1String, word2String):
    print "Yes"
else:
    print "No"


print "\n\nHolonym test:\n"

word1String= "molecule"
word2String= "atom"

print str(word1String) + " is Holonym of " + str(word2String) + "?"
if wn.isHolonym(word1String, word2String):
    print "Yes"
else:
    print "No"


print "\n\nAntonym test:\n"

word1String= "weak"
word2String= "strong"

print str(word1String) + " is Antonym of " + str(word2String) + "?"
if wn.isAntonym(word1String, word2String):
    print "Yes"
else:
    print "No"
"""


