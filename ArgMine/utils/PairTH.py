# -*- coding: utf-8 -*-

import postagger.Token as Token

class PairTH(object):
    '''
    Class representing a pair of texts from SICK or RTE.
    It is meant to be used as an abstract representation for both.
    '''
    def __init__(self, id, similarity, t, h, tTagged, hTagged):
        '''
        :param t: string with the text
        :param h: string with the hypothesis
        :param id_: int indicating id in the original file
        :param similarity: float
        '''
        self.t = t
        self.h = h
        self.tTagged = tTagged
        self.hTagged = hTagged
        self.id = id
        self.similarity = float(similarity)
    
    def __str__(self):
        outputString = "Pair (id= " + str(self.id) +  "; similarity= "+ str(self.similarity) + ")\n    t= " + str((self.t).encode("utf-8")) + "\n    h= " + str((self.h).encode("utf-8")) + "\n"
        
        outputString += "    tTagged= "
        for t in (self.tTagged):
            outputString += "(" + str(t) + ") "
            
        outputString += "\n"
        
        outputString += "    hTagged= "
        for t in (self.hTagged):
            outputString += "(" + str(t) + ") "
            
        outputString += "\n"
        
        return outputString
    
    