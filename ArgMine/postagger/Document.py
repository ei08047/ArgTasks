"""
Document: Corresponds to the complete document (in this case, news/opinion articles).
"""

class Document:
    
    def __init__(self, newsId, tokens, originalText):
        # set of tokens 
        self.tokens= tokens
        
        # original proposition (directly extracted from the text)
        self.originalText= originalText
        
        # newsId
        self.newsId= newsId
        
        self.propositionBoundaries_= []
        
        
    def __str__(self):
        intermediateOutput= ""
        
        for t in self.tokens:
            intermediateOutput= intermediateOutput + " " + str(t) + ","
        
        return "\n\nDocument id: " + str(self.newsId) + "\nset of tokens: \n" + intermediateOutput + "\n\n Original Text: \n" + self.originalText
    
    def setPropositionBoundaries(self, propositionBoundaries):
        self.propositionBoundaries_= propositionBoundaries
