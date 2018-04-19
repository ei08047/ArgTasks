#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Synset: 
"""

import copy
import os
import codecs

# paths

class Triple:
    
    def __init__(self, id1, id2, relationType, confidenceValue):
        # Document ID
        self.id1= id1
        
        self.id2= id2
        
        self.relationType= relationType
        
        self.confidenceValue= confidenceValue
        
    def __str__(self):
        
        stringOutput = "id1= " + str(self.id1) + " --> relationType= " + str(self.relationType) + " --> id2= " + str(self.id2) + " [" + str(self.confidenceValue) + "]"+ "\n"
        
        return stringOutput
        