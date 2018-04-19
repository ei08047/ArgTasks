#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Proposition: sentence that has a truth value. It is obtained by a POS Tagger tool -> sentence splitter. 
In this case, corresponds to a sequence of tokens and the original text.
"""

class Proposition:
    
    def __init__(self, tokens, originalText, newsId, absolutePositionInNews):
        # set of tokens 
        self.tokens= tokens
        
        # original proposition (directly extracted from the text)
        self.originalText= originalText
        
        # newsId where this Proposition come from
        self.newsId= newsId
        
        # Position of proposition absolutely in document (news)
        self.absolutePositionInNews= absolutePositionInNews
        
    
    def __str__(self):
        intermediateOutput= ""
        
        for t in self.tokens:
            intermediateOutput= intermediateOutput + " " + str(t) + ","
        
        return "\nset of tokens: \n" + intermediateOutput + "\n\n Original Text: \n" + self.originalText
    